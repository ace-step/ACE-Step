from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler
import uuid
import traceback

app = FastAPI(title="ACEStep Pipeline API")

class ACEStepInput(BaseModel):
    format: str
    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    actual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

def initialize_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int) -> ACEStepPipeline:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
    )

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    try:
        # Initialize pipeline
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )

        # Prepare parameters
        params = (
            input_data.format,
            input_data.audio_duration,
            input_data.prompt,
            input_data.lyrics,
            input_data.infer_step,
            input_data.guidance_scale,
            input_data.scheduler_type,
            input_data.cfg_type,
            input_data.omega_scale,
            ", ".join(map(str, input_data.actual_seeds)),
            input_data.guidance_interval,
            input_data.guidance_interval_decay,
            input_data.min_guidance_scale,
            input_data.use_erg_tag,
            input_data.use_erg_lyric,
            input_data.use_erg_diffusion,
            ", ".join(map(str, input_data.oss_steps)),
            input_data.guidance_scale_text,
            input_data.guidance_scale_lyric,
        )

        # Generate output path if not provided
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.wav"

        # Run pipeline
        model_demo(
            *params,
            save_path=output_path
        )

        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}\n{traceback.format_exc()}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
