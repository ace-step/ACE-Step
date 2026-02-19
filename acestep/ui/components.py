"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import gradio as gr
import librosa
import os


TAG_DEFAULT = "funk, pop, soul, rock, melodic, guitar, drums, bass, keyboard, percussion, 105 BPM, energetic, upbeat, groovy, vibrant, dynamic"
LYRIC_DEFAULT = """[verse]
Neon lights they flicker bright
City hums in dead of night
Rhythms pulse through concrete veins
Lost in echoes of refrains

[verse]
Bassline groovin' in my chest
Heartbeats match the city's zest
Electric whispers fill the air
Synthesized dreams everywhere

[chorus]
Turn it up and let it flow
Feel the fire let it grow
In this rhythm we belong
Hear the night sing out our song

[verse]
Guitar strings they start to weep
Wake the soul from silent sleep
Every note a story told
In this night we’re bold and gold

[bridge]
Voices blend in harmony
Lost in pure cacophony
Timeless echoes timeless cries
Soulful shouts beneath the skies

[verse]
Keyboard dances on the keys
Melodies on evening breeze
Catch the tune and hold it tight
In this moment we take flight
"""

# First, let's define the presets at the top of the file, after the imports
GENRE_PRESETS = {
    "Modern Pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant, female vocals, polished vocals",
    "Rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty, male vocals, raw vocals",
    "Hip Hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense, male vocals, rhythmic vocals",
    "Country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm, male vocals, twangy vocals",
    "EDM": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental",
    "Reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive, male vocals, smooth vocals",
    "Classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental",
    "Jazz": "jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful, male vocals, crooning vocals",
    "Metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy, male vocals, screamed vocals",
    "R&B": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic, female vocals, silky vocals"
}

# Add this function to handle preset selection
def update_tags_from_preset(preset_name):
    if preset_name == "Custom":
        return ""
    return GENRE_PRESETS.get(preset_name, "")


def create_output_ui(task_name="Text2Music"):
    # For many consumer-grade GPU devices, only one batch can be run
    output_audio1 = gr.Audio(type="filepath", label=f"{task_name} Generated Audio 1", interactive=False)
    # output_audio2 = gr.Audio(type="filepath", label="Generated Audio 2")
    with gr.Accordion(f"{task_name} Parameters", open=False):
        input_params_json = gr.JSON(label=f"{task_name} Parameters")
    # outputs = [output_audio1, output_audio2]
    outputs = output_audio1
    return outputs, input_params_json


def dump_func(*args, **kwargs):
    print()
    print(args)
    print(kwargs)
    return [None, {}]


def create_text2music_ui(
    text2music_process_func,
    sample_data_func=None,
    load_data_func=None,
):

    with gr.Row(equal_height=True):
        # Get base output directory from environment variable, defaulting to CWD-relative 'outputs'.
        # This default (./outputs) is suitable for non-Docker local development.
        # For Docker, the ACE_OUTPUT_DIR environment variable should be set (e.g., to /app/outputs).
        output_file_dir = os.environ.get("ACE_OUTPUT_DIR", "./outputs")
        if not os.path.isdir(output_file_dir):
            os.makedirs(output_file_dir, exist_ok=True)
        json_files = [f for f in os.listdir(output_file_dir) if f.endswith('.json')]
        json_files.sort(reverse=True, key=lambda x: int(x.split('_')[1]))
        output_files = gr.Dropdown(choices=json_files, label="Select previous generated input params", scale=9, )
        load_bnt = gr.Button("Load", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                # add markdown, tags and lyrics examples are from ai music generation community
                audio_duration = gr.Slider(
                    -1,
                    240.0,
                    step=0.00001,
                    value=-1,
                    label="Audio Duration",
                    info="-1 means random duration (30 ~ 240).",
                    scale=9,
                )
                format = gr.Dropdown(choices=["mp3", "ogg", "flac", "wav"], value="wav", label="Format") # FIXME: recommend rename to _format
                sample_bnt = gr.Button("Sample", variant="secondary", scale=1)

            # audio2audio
            with gr.Row(equal_height=True):
                audio2audio_enable = gr.Checkbox(label="Enable Audio2Audio", value=False, info="Check to enable Audio-to-Audio generation using a reference audio.", elem_id="audio2audio_checkbox")
                lora_name_or_path = gr.Dropdown(
                    label="Lora Name or Path",
                    choices=["ACE-Step/ACE-Step-v1-chinese-rap-LoRA", "none"],
                    value="none",
                    allow_custom_value=True,
                    min_width=300
                )
                lora_weight = gr.Number(value=1.0, label="Lora weight", step=0.1, maximum=3, minimum=-3)

            ref_audio_input = gr.Audio(type="filepath", label="Reference Audio (for Audio2Audio)", visible=False, elem_id="ref_audio_input", show_download_button=True)
            ref_audio_strength = gr.Slider(
                label="Refer audio strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                elem_id="ref_audio_strength",
                visible=False,
            )

            def toggle_ref_audio_visibility(is_checked):
                return (
                    gr.update(visible=is_checked, elem_id="ref_audio_input"),
                    gr.update(visible=is_checked, elem_id="ref_audio_strength"),
                )

            audio2audio_enable.change(
                fn=toggle_ref_audio_visibility,
                inputs=[audio2audio_enable],
                outputs=[ref_audio_input, ref_audio_strength],
            )

            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("""<center>Support tags, descriptions, and scene. Use commas to separate different tags.<br>Tags and lyrics examples are from AI music generation community.</center>""")
                    with gr.Row():
                        genre_preset = gr.Dropdown(
                            choices=["Custom"] + list(GENRE_PRESETS.keys()),
                            value="Custom",
                            label="Preset",
                            scale=1,
                        )
                        prompt = gr.Textbox(
                            lines=1,
                            label="Tags",
                            max_lines=4,
                            value=TAG_DEFAULT,
                            scale=9,
                        )

            # Add the change event for the preset dropdown
            genre_preset.change(
                fn=update_tags_from_preset,
                inputs=[genre_preset],
                outputs=[prompt]
            )
            with gr.Group():
                gr.Markdown("""<center>Support lyric structure tags like [verse], [chorus], and [bridge] to separate different parts of the lyrics.<br>Use [instrumental] or [inst] to generate instrumental music. Not support genre structure tag in lyrics</center>""")
                lyrics = gr.Textbox(
                    lines=9,
                    label="Lyrics",
                    max_lines=13,
                    value=LYRIC_DEFAULT,
                )

            with gr.Accordion("Basic Settings", open=False):
                infer_step = gr.Slider(
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=60,
                    label="Infer Steps",
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    value=15.0,
                    label="Guidance Scale",
                    info="When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied.",
                )
                guidance_scale_text = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale Text",
                    info="Guidance scale for text condition. It can only apply to cfg. set guidance_scale_text=5.0, guidance_scale_lyric=1.5 for start",
                )
                guidance_scale_lyric = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale Lyric",
                )

                manual_seeds = gr.Textbox(
                    label="manual seeds (default None)",
                    placeholder="1,2,3,4",
                    value=None,
                    info="Seed for the generation",
                )

            with gr.Accordion("Advanced Settings", open=False):
                scheduler_type = gr.Radio(
                    ["euler", "heun", "pingpong"],
                    value="euler",
                    label="Scheduler Type",
                    elem_id="scheduler_type",
                    info="Scheduler type for the generation. euler is recommended. heun will take more time. pingpong use SDE",
                )
                cfg_type = gr.Radio(
                    ["cfg", "apg", "cfg_star"],
                    value="apg",
                    label="CFG Type",
                    elem_id="cfg_type",
                    info="CFG type for the generation. apg is recommended. cfg and cfg_star are almost the same.",
                )
                use_erg_tag = gr.Checkbox(
                    label="use ERG for tag",
                    value=True,
                    info="Use Entropy Rectifying Guidance for tag. It will multiple a temperature to the attention to make a weaker tag condition and make better diversity.",
                )
                use_erg_lyric = gr.Checkbox(
                    label="use ERG for lyric",
                    value=False,
                    info="The same but apply to lyric encoder's attention.",
                )
                use_erg_diffusion = gr.Checkbox(
                    label="use ERG for diffusion",
                    value=True,
                    info="The same but apply to diffusion model's attention.",
                )

                omega_scale = gr.Slider(
                    minimum=-100.0,
                    maximum=100.0,
                    step=0.1,
                    value=10.0,
                    label="Granularity Scale",
                    info="Granularity scale for the generation. Higher values can reduce artifacts",
                )

                guidance_interval = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Guidance Interval",
                    info="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)",
                )
                guidance_interval_decay = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    label="Guidance Interval Decay",
                    info="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay.",
                )
                min_guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=200.0,
                    step=0.1,
                    value=3.0,
                    label="Min Guidance Scale",
                    info="Min guidance scale for guidance interval decay's end scale",
                )
                oss_steps = gr.Textbox(
                    label="OSS Steps",
                    placeholder="16, 29, 52, 96, 129, 158, 172, 183, 189, 200",
                    value=None,
                    info="Optimal Steps for the generation. But not test well",
                )

            text2music_bnt = gr.Button("Generate", variant="primary")

        with gr.Column():
            outputs, input_params_json = create_output_ui()

            for i in range(2):
                gr.Markdown('# ' + chr(0x200e)) # ToDo: find better way to space these out

            retake_variance = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="variance"
            )
            retake_seeds = gr.Textbox(
                label="retake seeds (default None)", placeholder="", value=None
            )
            with gr.Tabs() as text2music_tabs:
                def toggle_variance_interact(event_data: gr.SelectData):
                    not_edit = event_data.value != 'edit'
                    return [
                        gr.update(interactive=not_edit),
                        gr.update(interactive=not_edit),
                    ]
                text2music_tabs.select(
                    fn=toggle_variance_interact,
                    outputs=[retake_seeds, retake_variance]
                )
                with gr.Tab("retake"):
                    retake_bnt = gr.Button("Retake", variant="primary")
                    retake_outputs, retake_input_params_json = create_output_ui("Retake")

                with gr.Tab("repainting"):
                    repaint_start = gr.Slider(
                        minimum=0.0,
                        maximum=240.0,
                        step=0.01,
                        value=0.0,
                        label="Repaint Start Time",
                    )
                    repaint_end = gr.Slider(
                        minimum=0.0,
                        maximum=240.0,
                        step=0.01,
                        value=30.0,
                        label="Repaint End Time",
                    )
                    repaint_source = gr.Radio(
                        ["text2music", "last_repaint", "upload"],
                        value="text2music",
                        label="Repaint Source",
                        elem_id="repaint_source",
                    )

                    repaint_source_audio_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="repaint_source_audio_upload",
                        show_download_button=True,
                    )
                    repaint_source.change(
                        fn=lambda x: gr.update(
                            visible=x == "upload", elem_id="repaint_source_audio_upload"
                        ),
                        inputs=[repaint_source],
                        outputs=[repaint_source_audio_upload],
                    )

                    repaint_bnt = gr.Button("Repaint", variant="primary")
                    repaint_outputs, repaint_input_params_json = create_output_ui("Repaint")

                with gr.Tab("edit"):
                    edit_target_prompt = gr.Textbox(lines=2, label="Edit Tags", max_lines=4)
                    edit_target_lyrics = gr.Textbox(lines=9, label="Edit Lyrics", max_lines=13)

                    edit_type = gr.Radio(
                        ["only_lyrics", "remix"],
                        value="only_lyrics",
                        label="Edit Type",
                        elem_id="edit_type",
                        info="`only_lyrics` will keep the whole song the same except lyrics difference. Make your diffrence smaller, e.g. one lyrc line change.\nremix can change the song melody and genre",
                    )
                    edit_n_min = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.6,
                        label="edit_n_min",
                    )
                    edit_n_max = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=1.0,
                        label="edit_n_max",
                    )

                    def edit_type_change_func(edit_type):
                        if edit_type == "only_lyrics":
                            n_min = 0.6
                            n_max = 1.0
                        elif edit_type == "remix":
                            n_min = 0.2
                            n_max = 0.4
                        return n_min, n_max

                    edit_type.change(
                        edit_type_change_func,
                        inputs=[edit_type],
                        outputs=[edit_n_min, edit_n_max],
                    )

                    edit_source = gr.Radio(
                        ["text2music", "last_edit", "upload"],
                        value="text2music",
                        label="Edit Source",
                        elem_id="edit_source",
                    )
                    edit_source_audio_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="edit_source_audio_upload",
                        show_download_button=True,
                    )
                    edit_source.change(
                        fn=lambda x: gr.update(
                            visible=x == "upload", elem_id="edit_source_audio_upload"
                        ),
                        inputs=[edit_source],
                        outputs=[edit_source_audio_upload],
                    )

                    edit_bnt = gr.Button("Edit", variant="primary")
                    edit_outputs, edit_input_params_json = create_output_ui("Edit")

                with gr.Tab("extend"):
                    extend_seeds = gr.Textbox(
                        label="extend seeds (default None)", placeholder="", value=None
                    )
                    left_extend_length = gr.Slider(
                        minimum=0.0,
                        maximum=240.0,
                        step=0.01,
                        value=0.0,
                        label="Left Extend Length",
                    )
                    right_extend_length = gr.Slider(
                        minimum=0.0,
                        maximum=240.0,
                        step=0.01,
                        value=30.0,
                        label="Right Extend Length",
                    )
                    extend_source = gr.Radio(
                        ["text2music", "last_extend", "upload"],
                        value="text2music",
                        label="Extend Source",
                        elem_id="extend_source",
                    )

                    extend_source_audio_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="extend_source_audio_upload",
                        show_download_button=True,
                    )
                    extend_source.change(
                        fn=lambda x: gr.update(
                            visible=x == "upload", elem_id="extend_source_audio_upload"
                        ),
                        inputs=[extend_source],
                        outputs=[extend_source_audio_upload],
                    )

                    extend_bnt = gr.Button("Extend", variant="primary")
                    extend_outputs, extend_input_params_json = create_output_ui("Extend")

    all_gradio_component_names = []
    all_gradio_components = []
    l = locals()
    for k in l:
        var = l[k]
        if isinstance(var, gr.components.Component):
            all_gradio_components.append(var)
            all_gradio_component_names.append(k)

    def get_kwargs(args):
        assert len(args) == len(all_gradio_component_names), 'name list length != argument list length'
        kwargs = {}
        for i in range(len(args)):
            kwargs[all_gradio_component_names[i]] = args[i]
        return kwargs

    text2music_outputs = [
        audio_duration,
        prompt,
        lyrics,
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_tag,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
        audio2audio_enable,
        ref_audio_strength,
        ref_audio_input,

        lora_name_or_path,
        lora_weight,
        #retake_variance,
        #retake_seeds,
    ]
    l = locals()
    lk = list(l.keys())
    lv = list(l.values())
    text2music_output_names = [ lk[lv.index(thing)] for thing in text2music_outputs ]

    def _get_args(kwargs):
        args = []
        for name, component in zip(text2music_output_names, text2music_outputs):
            v = kwargs.get(name, None)
            if isinstance(component, gr.components.Textbox) and isinstance(v, list):
                v = str(v)[1:-1] # remove brackets
            args.append(v)
        return args

    def select_input(kwargs, edit_source, upload_path, last_json):
        params = {}
        if kwargs[edit_source] == "upload":
            params['src_audio_path'] = kwargs[upload_path]
            params['audio_duration'] = librosa.get_duration(path=kwargs[upload_path])
        elif kwargs[edit_source] == "text2music":
            params.update(kwargs['input_params_json'])
            params['src_audio_path'] = params['audio_path']
        else:
            params.update(kwargs[last_json])
            params['src_audio_path'] = params['audio_path']

        kwargs['src_audio_path'] = params['src_audio_path']
        kwargs['audio_duration'] = params['audio_duration']
        return kwargs

    def sample_data(*args):
        kwargs = get_kwargs(args)
        json_data = sample_data_func(kwargs['lora_name_or_path'])
        json_data['manual_seeds'] = json_data['actual_seeds']
        return _get_args(json_data)

    def load_data(*args):
        kwargs = get_kwargs(args)
        if isinstance(output_file_dir, str):
            json_file = os.path.join(output_file_dir, kwargs['output_files'])
        json_data = load_data_func(json_file)
        json_data['manual_seeds'] = json_data['actual_seeds']
        return _get_args(json_data)

    def generate_process_function(*args):
        kwargs = get_kwargs(args)
        kwargs['task'] = 'text2music'
        return text2music_process_func(**kwargs)

    def retake_process_func(*args):
        kwargs = get_kwargs(args)
        kwargs['manual_seeds'] = kwargs['input_params_json']['actual_seeds']
        kwargs['task'] = 'retake'
        return text2music_process_func(**kwargs)

    def edit_process_func(*args):
        kwargs = get_kwargs(args)
        kwargs = select_input(kwargs, 'edit_source', 'edit_source_audio_upload', 'edit_input_params_json')

        if not kwargs.get('edit_target_prompt', ''):
            kwargs['edit_target_prompt'] = kwargs['prompt']
        if not kwargs.get('edit_target_lyrics', ''):
            kwargs['edit_target_lyrics'] = kwargs['lyrics']

        kwargs['task'] = 'edit'
        return text2music_process_func(**kwargs)

    def extend_process_func(*args):
        kwargs = get_kwargs(args)
        kwargs = select_input(kwargs, 'extend_source', 'extend_source_audio_upload', 'extend_input_params_json')

        kwargs['repaint_start'] = -kwargs['left_extend_length']
        kwargs['repaint_end'] = kwargs['audio_duration'] + kwargs['right_extend_length']
        kwargs['audio_duration'] += kwargs['right_extend_length'] + kwargs['left_extend_length']
        kwargs['task'] = 'extend'
        return text2music_process_func(**kwargs)

    def repaint_process_func(*args):
        kwargs = get_kwargs(args)
        kwargs = select_input(kwargs, 'repaint_source', 'repaint_source_audio_upload', 'repaint_input_params_json')

        kwargs['task'] = 'repaint'
        return text2music_process_func(**kwargs)

    text2music_bnt.click(
        generate_process_function,
        inputs=all_gradio_components,
        outputs=[outputs, input_params_json]
    )
    retake_bnt.click(
        retake_process_func,
        inputs=all_gradio_components,
        outputs=[retake_outputs, retake_input_params_json]
    )
    repaint_bnt.click(
        repaint_process_func,
        inputs=all_gradio_components,
        outputs=[repaint_outputs, repaint_input_params_json]
    )
    edit_bnt.click(
        edit_process_func,
        inputs=all_gradio_components,
        outputs=[edit_outputs, edit_input_params_json]
    )
    extend_bnt.click(
        extend_process_func,
        inputs=all_gradio_components,
        outputs=[extend_outputs, extend_input_params_json]
    )
    sample_bnt.click(
        sample_data,
        inputs=all_gradio_components,
        outputs=text2music_outputs
    )
    load_bnt.click(
        load_data,
        inputs=all_gradio_components,
        outputs=text2music_outputs
    )

def create_main_demo_ui(
    text2music_process_func=dump_func,
    sample_data_func=dump_func,
    load_data_func=dump_func,
):
    with gr.Blocks(title="ACE-Step Model 1.0 DEMO") as demo:
        gr.Markdown(
            """
                <h1 style="text-align: center;">ACE-Step: A Step Towards Music Generation Foundation Model</h1>
            """
        )
        with gr.Tab("text2music"):
            create_text2music_ui(
                text2music_process_func=text2music_process_func,
                sample_data_func=sample_data_func,
                load_data_func=load_data_func,
            )
    return demo


if __name__ == "__main__":
    demo = create_main_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
