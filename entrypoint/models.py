from typing import Optional, List
from pydantic import BaseModel, model_validator, Field, ValidationError

class S3Config(BaseModel):
    bucket: str
    prefix_path: str
    aws_access_key_id: str
    aws_secret_access_key: str
    
class PreprocessRequest(BaseModel):
    video_path: str  # S3 URL
    refer_path: str  # S3 URL
    resolution_area: List[int]
    retarget_flag: Optional[bool] = False
    use_flux: Optional[bool] = False
    iterations: Optional[int] = None
    k: Optional[int] = None
    w_len: Optional[int] = None
    h_len: Optional[int] = None
    replace_flag: Optional[bool] = False

    @model_validator(mode='after')
    def check_mutually_exclusive_flags(self):
        if self.replace_flag and self.use_flux:
            raise ValueError('`replace_flag` and `use_flux` cannot be True simultaneously.')
        return self

    @model_validator(mode='after')
    def check_conditional_fields(self):
        if self.replace_flag:
            if self.iterations is None:
                raise ValueError('`iterations` is required when `replace_flag` is True.')
            if self.k is None:
                raise ValueError('`k` is required when `replace_flag` is True.')
            if self.w_len is None:
                raise ValueError('`w_len` is required when `replace_flag` is True.')
            if self.h_len is None:
                raise ValueError('`h_len` is required when `replace_flag` is True.')
        
        if self.use_flux:
            if not self.retarget_flag:
                raise ValueError('`retarget_flag` must be True when `use_flux` is True.')
        return self

class OutputFile(BaseModel):
    file: str
    url: str

class PreprocessResponse(BaseModel):
    message: str
    output_s3_paths: List[OutputFile]
    file_prefix: str

class AppConfig(BaseModel):
    s3_config: S3Config
    preprocess_ckpt_path: str
    inference_ckpt_path: str

class GenerateRequest(BaseModel):
    task: str = Field(..., description="Task name, e.g., animate-14B")
    src_root_path: str = Field(..., description="S3 folder URL for source data")
    refert_num: int = Field(1, description="Reference number")
    replace_flag: Optional[bool] = False
    use_relighting_lora: Optional[bool] = False
    nproc_per_node: int = Field(1, description="Number of processes per node for distributed training")
    nnodes: int = Field(1, description="Number of nodes for distributed training")
    dit_fsdp: Optional[bool] = False
    t5_fsdp: Optional[bool] = False
    ulysses_size: Optional[int] = None

    @model_validator(mode='after')
    def check_conditional_generate_fields(self):
        if self.replace_flag:
            if not self.use_relighting_lora:
                raise ValueError('`use_relighting_lora` is required when `replace_flag` is True for generation.')
        
        if self.nproc_per_node > 1:
            if not self.dit_fsdp or not self.t5_fsdp or self.ulysses_size is None:
                raise ValueError('`dit_fsdp`, `t5_fsdp`, and `ulysses_size` are required for multi-card generation.')
        return self

class GenerateResponse(BaseModel):
    message: str
    output_s3_paths: List[OutputFile]
