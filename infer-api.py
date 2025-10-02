from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler
import uuid

app = FastAPI(title="ACEStep Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # exact scheme+host+port
    allow_credentials=False,                  # omit or set False if you don't need cookies/auth
    allow_methods=["*"],
    allow_headers=["*"],
)

class ACEStepInput(BaseModel):
    prompt: str
    lyrics: Optional[str] = None
    checkpoint_path: str = "ACE-Step/ACE-Step-v1-3.5B"
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float = 60.0
    infer_step: int = 27
    guidance_scale: float = 3.5
    scheduler_type: str = "euler"
    cfg_type: str = "cfg"
    omega_scale: float = 1.0
    actual_seeds: List[int] = Field(default_factory=list)
    guidance_interval: float = 1.0
    guidance_interval_decay: float = 0.0
    min_guidance_scale: float = 1.0
    use_erg_tag: bool = False
    use_erg_lyric: bool = False
    use_erg_diffusion: bool = False
    oss_steps: List[int] = Field(default_factory=list)
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
async def generate_audio(
        # core text fields (form)
        prompt: str = Form(...),
        lyrics: Optional[str] = Form(None),
        # model/runtime knobs (form with defaults)
        checkpoint_path: str = Form("ACE-Step/ACE-Step-v1-3.5B"),
        bf16: bool = Form(True),
        torch_compile: bool = Form(False),
        device_id: int = Form(0),
        output_path: Optional[str] = Form(None),
        audio_duration: float = Form(60.0),
        infer_step: int = Form(27),
        guidance_scale: float = Form(3.5),
        scheduler_type: str = Form("euler"),
        cfg_type: str = Form("cfg"),
        omega_scale: float = Form(1.0),
        actual_seeds: Optional[str] = Form(None),   # e.g. "42,7"
        guidance_interval: float = Form(1.0),
        guidance_interval_decay: float = Form(0.0),
        min_guidance_scale: float = Form(1.0),
        use_erg_tag: bool = Form(False),
        use_erg_lyric: bool = Form(False),
        use_erg_diffusion: bool = Form(True),
        oss_steps: Optional[str] = Form(None),      # e.g. "10,20,40"
        guidance_scale_text: float = Form(0.0),
        guidance_scale_lyric: float = Form(0.0),
        # optional file upload via multipart/form-data
        audio_file: UploadFile | None = File(None),
    ):
    try:
        # Initialize pipeline
        model_demo = initialize_pipeline(
            checkpoint_path,
            bf16,
            torch_compile,
            device_id
        )
        # Convert/normalize form fields
        manual_seeds = (
            [int(s) for s in actual_seeds.split(",") if s.strip()] if actual_seeds else None
        )
        oss_steps_str = oss_steps or ""
        lyrics = "" if lyrics is None else str(lyrics)
 
        if audio_file:
            tmp_ref = f"/tmp/ref_{uuid.uuid4().hex}_{audio_file.filename}"
            with open(tmp_ref, "wb") as f:
                f.write(await audio_file.read())

        # Generate output path if not provided
        output_path = output_path or f"/tmp/output_{uuid.uuid4().hex}.wav"

        model_demo(
            audio_duration=float(audio_duration),
            prompt=str(prompt),
            lyrics=lyrics,
            infer_step=int(infer_step),
            guidance_scale=float(guidance_scale),
            scheduler_type=str(scheduler_type),
            cfg_type=str(cfg_type),
            omega_scale=float(omega_scale),
            manual_seeds=manual_seeds,
            guidance_interval=float(guidance_interval),
            guidance_interval_decay=float(guidance_interval_decay),
            min_guidance_scale=float(min_guidance_scale),
            use_erg_tag=bool(use_erg_tag),
            use_erg_lyric=bool(use_erg_lyric),
            use_erg_diffusion=bool(use_erg_diffusion),
            oss_steps=oss_steps_str,
            guidance_scale_text=float(guidance_scale_text),
            guidance_scale_lyric=float(guidance_scale_lyric),
            save_path=output_path,
        )


        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
