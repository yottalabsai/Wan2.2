import os
import subprocess
import tempfile
import shutil
import uuid
import logging # Import logging module
import sys # Import sys module
import asyncio
import time

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from entrypoint.models import PreprocessRequest, PreprocessResponse, S3Config, OutputFile, AppConfig, GenerateRequest, GenerateResponse # Import OutputFile, AppConfig, GenerateRequest, GenerateResponse
from entrypoint.s3_util import download_s3_url_to_temp, get_s3_client, upload_file_and_get_presigned_url, download_s3_folder_to_temp # Import download_s3_folder_to_temp
from entrypoint.log import setup_logging # Import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)

# Global lock for request processing
_processing_lock = asyncio.Lock()
LOCK_TIMEOUT = 1  # 1 second timeout


def read_config():
    load_dotenv()
    bucket = os.getenv("S3_BUCKET")
    prefix_path = os.getenv("S3_PREFIX_PATH")
    aws_access_key_id = os.getenv("S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    preprocess_ckpt_path = os.getenv("PREPROCESS_CKPT_PATH") # Read preprocess_ckpt_path
    inference_ckpt_path = os.getenv("INFERENCE_CKPT_PATH") # Read inference_ckpt_path

    # Non-empty checks for environment variables
    if not bucket:
        raise ValueError("S3_BUCKET environment variable must be set and not empty.")
    if not prefix_path:
        raise ValueError("S3_PREFIX_PATH environment variable must be set and not empty.")
    if not aws_access_key_id:
        raise ValueError("S3_AWS_ACCESS_KEY_ID environment variable must be set and not empty.")
    if not aws_secret_access_key:
        raise ValueError("S3_AWS_SECRET_ACCESS_KEY environment variable must be set and not empty.")
    if not preprocess_ckpt_path:
        raise ValueError("PREPROCESS_CKPT_PATH environment variable must be set and not empty.")
    if not inference_ckpt_path:
        raise ValueError("INFERENCE_CKPT_PATH environment variable must be set and not empty.")

    s3 = S3Config(
        bucket=bucket,
        prefix_path=prefix_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    return AppConfig(
        s3_config=s3,
        preprocess_ckpt_path=preprocess_ckpt_path,
        inference_ckpt_path=inference_ckpt_path
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging at application startup
    setup_logging()
    logger.info("FastAPI application starting up.")

    # Initialize AppConfig from environment variables
    app_config = read_config()
    app.state.s3_config = app_config.s3_config
    app.state.s3_client = get_s3_client(app.state.s3_config)
    app.state.preprocess_ckpt_path = app_config.preprocess_ckpt_path
    app.state.inference_ckpt_path = app_config.inference_ckpt_path
    yield
    # Clean up resources on shutdown
    logger.info("FastAPI application is shutting down. Resources can be cleaned up here.")

app = FastAPI(lifespan=lifespan)

async def run_subprocess_and_stream_output_async(command, env, cwd):
    """
    Run a subprocess and stream its output line by line using asyncio.
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        
        output = line.decode('utf-8')
        yield output
        
    return_code = await process.wait()
    yield f"Process completed with return code: {return_code}\n"


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_video(request: PreprocessRequest):
    # Try to acquire the lock with a timeout
    try:
        acquired = await asyncio.wait_for(_processing_lock.acquire(), timeout=LOCK_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="System busy, please try again later")

    temp_dir = None
    try:
        s3_client = app.state.s3_client
        ckpt_path = app.state.preprocess_ckpt_path # Get preprocess_ckpt_path from app.state

        # Create temporary directory for downloads and outputs
        temp_dir = tempfile.mkdtemp()
        unique_id = str(uuid.uuid4())

        # Use download_s3_url_to_temp for video and reference image
        local_video_path = download_s3_url_to_temp(s3_client, request.video_path)
        if not local_video_path:
            raise HTTPException(status_code=500, detail=f"Failed to download video from {request.video_path}")

        local_refer_path = download_s3_url_to_temp(s3_client, request.refer_path)
        if not local_refer_path:
            raise HTTPException(status_code=500, detail=f"Failed to download reference image from {request.refer_path}")

        local_save_path = os.path.join(temp_dir, f"{unique_id}_process_results")
        os.makedirs(local_save_path, exist_ok=True)

        # 2. Construct the command based on flags
        command = [
            sys.executable, # Use sys.executable to ensure the correct python interpreter is used
            "wan/modules/animate/preprocess/preprocess_data.py", # Corrected path relative to project root
            "--ckpt_path", ckpt_path,
            "--video_path", local_video_path,
            "--refer_path", local_refer_path,
            "--save_path", local_save_path,
            "--resolution_area", str(request.resolution_area[0]), str(request.resolution_area[1]),
        ]

        if request.replace_flag:
            command.extend([
                "--iterations", str(request.iterations),
                "--k", str(request.k),
                "--w_len", str(request.w_len),
                "--h_len", str(request.h_len),
                "--replace_flag"
            ])
        elif request.use_flux: # Mutually exclusive check already in model
            command.extend([
                "--retarget_flag", # Assuming retarget_flag is always true with use_flux based on user's example
                "--use_flux"
            ])

        # Get current environment and update PYTHONPATH and PYTHON
        env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in env.get('PYTHONPATH', '').split(os.pathsep):
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [project_root, env.get('PYTHONPATH', '')]))
        
        # Ensure 'python' command within torch.distributed.run uses the correct interpreter
        env['PYTHON'] = sys.executable

        logger.info(f"Executing preprocessing command: {' '.join(command)}")
        
        # 3. Execute the command and stream output
        return_code = 0
        async for output in run_subprocess_and_stream_output_async(command, env, project_root):
            logger.info(output)
            if "Process completed with return code:" in output:
                try:
                    return_code = int(output.split(":")[-1].strip())
                except ValueError:
                    logger.error(f"Could not parse return code from: {output}")
                    return_code = 1 # 视为失败

        if return_code != 0:
            logger.error(f"Script failed with exit code {return_code}")
            raise HTTPException(status_code=500, detail=f"Script failed with exit code {return_code}")

        # 4. Upload results to S3
        uploaded_files = []
        # Construct the S3 object prefix for this specific preprocess run
        s3_upload_prefix = f"{app.state.s3_config.prefix_path.rstrip('/')}/preprocess_result/{unique_id}"
        for root, _, files in os.walk(local_save_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                # The upload_file_and_get_presigned_url function handles S3 key construction internally
                # and also deletes the local file after upload.
                presigned_url = upload_file_and_get_presigned_url(s3_client, app.state.s3_config, local_file_path, s3_object_prefix=s3_upload_prefix)
                if presigned_url:
                    # The file name in OutputFile should reflect the S3 key, not just the local basename
                    s3_key_suffix = os.path.relpath(local_file_path, local_save_path)
                    uploaded_files.append(OutputFile(file=f"preprocess_result/{unique_id}/{s3_key_suffix}", url=presigned_url))
                else:
                    logger.warning(f"Failed to upload {local_file_path} or get presigned URL.")

        return PreprocessResponse(message="Preprocessing completed successfully", output_s3_paths=uploaded_files, file_prefix=s3_upload_prefix)

    except Exception as e:
        logger.error(f"Error in preprocess_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Release the lock if we acquired it
        if _processing_lock.locked():
            _processing_lock.release()

@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest):
    # Try to acquire the lock with a timeout
    try:
        acquired = await asyncio.wait_for(_processing_lock.acquire(), timeout=LOCK_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="System busy, please try again later")

    temp_dir = None
    try:
        s3_client = app.state.s3_client
        ckpt_dir = app.state.inference_ckpt_path # Get inference_ckpt_path from app.state

        temp_dir = tempfile.mkdtemp()
        unique_id = str(uuid.uuid4())
        
        # 1. Download S3 folder contents to a local temporary directory
        local_src_root_path = download_s3_folder_to_temp(s3_client, request.src_root_path)
        if not local_src_root_path:
            raise HTTPException(status_code=500, detail=f"Failed to download source folder from {request.src_root_path}")

        # local_save_path should be a file, not a directory
        local_output_file = os.path.join(temp_dir, f"{unique_id}_output.mp4") # Assuming .mp4 output

        # 2. Construct the command based on flags and nproc_per_node
        base_command = [
            sys.executable, # Use sys.executable to ensure the correct python interpreter is used
            "generate.py", # Corrected path relative to project root
            "--task", request.task,
            "--ckpt_dir", ckpt_dir,
            "--src_root_path", local_src_root_path,
            "--refert_num", str(request.refert_num),
            "--save_file", local_output_file # Changed to --save_file
        ]

        if request.replace_flag:
            base_command.extend([
                "--replace_flag",
                "--use_relighting_lora"
            ])
        
        command = []
        if request.nproc_per_node > 1:
            command.extend([
                sys.executable, "-m", "torch.distributed.run", # Use sys.executable here as well
                "--nnodes", str(request.nnodes),
                "--nproc_per_node", str(request.nproc_per_node)
            ])
            # Append the script name and its arguments from base_command,
            # skipping the first element (sys.executable) of base_command
            command.extend(base_command[1:])
            if request.dit_fsdp:
                command.append("--dit_fsdp")
            if request.t5_fsdp:
                command.append("--t5_fsdp")
            if request.ulysses_size is not None:
                command.extend(["--ulysses_size", str(request.ulysses_size)])
        else:
            command.extend(base_command)

        
        
        # Get current environment and update PYTHONPATH and PYTHON
        env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in env.get('PYTHONPATH', '').split(os.pathsep):
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [project_root, env.get('PYTHONPATH', '')]))
        
        # Ensure 'python' command within torch.distributed.run uses the correct interpreter
        env['PYTHON'] = sys.executable

        # 3. Execute the command and stream output
        logger.info(f"Executing generation command: {' '.join(command)}")
        return_code = 0
        async for output in run_subprocess_and_stream_output_async(command, env, project_root):
            logger.info(output)
            if "Process completed with return code:" in output:
                try:
                    return_code = int(output.split(":")[-1].strip())
                except ValueError:
                    logger.error(f"Could not parse return code from: {output}")
                    return_code = 1 # 视为失败

        if return_code != 0:
            logger.error(f"Script failed with exit code {return_code}")
            raise HTTPException(status_code=500, detail=f"Script failed with exit code {return_code}")
        
        # 4. Upload results to S3 (single file upload)
        uploaded_files = []
        # Construct the S3 object prefix for this specific preprocess run
        s3_upload_prefix = f"{app.state.s3_config.prefix_path.rstrip('/')}/generate_video/{unique_id}"
        if os.path.exists(local_output_file):
            presigned_url = upload_file_and_get_presigned_url(s3_client, app.state.s3_config, local_output_file, s3_upload_prefix)
            if presigned_url:
                uploaded_files.append(OutputFile(file=os.path.basename(local_output_file), url=presigned_url))
            else:
                logger.warning(f"Failed to upload {local_output_file} or get presigned URL.")
        else:
            logger.error(f"Generated output file not found: {local_output_file}")
            raise HTTPException(status_code=500, detail=f"Generated output file not found: {local_output_file}")

        return GenerateResponse(message="Generation completed successfully", output_s3_paths=uploaded_files)

    except Exception as e:
        logger.error(f"Error in generate_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Release the lock if we acquired it
        if _processing_lock.locked():
            _processing_lock.release()