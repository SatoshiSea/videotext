import re
import uuid
import json
from pydub import AudioSegment
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import requests
import time
import subprocess
from pathlib import Path
import os
import pandas as pd
import wave
import json
from vosk import Model, KaldiRecognizer

# Definir la ruta base utilizando pathlib
base_path = Path(__file__).parent.resolve()
bin_path = base_path / "bin" / "ffmpeg"
model_path = base_path / "vosk-model-small-en-us-0.15"
cores = os.cpu_count()  

# Crear rutas para cada tipo de archivo
video_folder = base_path / "videos"
audio_folder = base_path / "audios"
transcription_folder = base_path / "transcriptions"
image_folder = base_path / "images"

# Crear carpetas si no existen
for folder in [video_folder, audio_folder, transcription_folder, image_folder]:
    if not folder.exists():
        folder.mkdir()
        print(f"Carpeta creada en: {folder}")
    else:
        print(f"Carpeta ya existe en: {folder}")

VOICE_ID = "pNInz6obpgDQGcFmaJgB" 
FONT_NAME = "Trends"
ELEVENLABS_API_KEY = "sk_8d4fe56dc35af3cda6f4c25fa557f59a5fa8828755684c32"
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

url_api = "https://api.dezgo.com/"
api_key = "DEZGO-EF5AF98400D97B80A918240CBF4A5DFC08444CE2B66F58414F509A6D429293351D57FD46"
api_endpoint = "text2image_flux"
api_width = 1280  # 720
api_height = 720  # 1280
api_sampler = "auto"
api_model_id = "juggernautxl_9_lightning_1024px"
api_negative_prompt = ""
api_seed = ""
api_format = "jpg"
api_guidance = 2
api_transparent_background = False

# General variables and function
fps = 30  # 25 or 3
resolution = "1920x1080"  # '1080X1920', '1920x1080','1280x720','854x480','640x360'
aspect_ratio = "16:9"  # '16:9','4:3','1:1','9:16'
video_bitrate = "3000k"  # '1000k','2000k','3000k','4000k'

def setup_fontconfig(base_path: Path):
    
    fonts_folder = base_path / "fonts"
    if not fonts_folder.exists():
        fonts_folder.mkdir()
        print(f"Carpeta de fuentes creada en: {fonts_folder}")
    else:
        print(f"Carpeta de fuentes ya existe en: {fonts_folder}")


    fonts_conf_path = base_path / "fonts.conf"
    fonts_conf_content = """<?xml version=\"1.0\"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
    <dir>{}</dir>
    <dir>C:\\Windows\\Fonts</dir>
</fontconfig>
""".format(fonts_folder.resolve())

    with open(fonts_conf_path, "w") as fonts_conf_file:
        fonts_conf_file.write(fonts_conf_content)

    os.environ["FONTCONFIG_FILE"] = str(fonts_conf_path)

def convert_audio_to_wav(audio_file, output_folder):
    audio = AudioSegment.from_mp3(audio_file)
    output_wav = output_folder / "output.wav"
    audio.export(output_wav, format="wav", parameters=["-ar", "16000"])
    return output_wav

def create_images_ia(api_key, url_api, api_endpoint, api_prompt, api_width, api_height, api_sampler, api_model_id, api_negative_prompt, api_seed, api_format, api_guidance, api_transparent_background, api_execution, image_folder_path, retry_delay=5, max_retries=3):
    url = f"{url_api}/{api_endpoint}"
    headers = {
        'X-Dezgo-Key': api_key
    }

    prompts = api_prompt.split('|')
    num_prompts = len(prompts)

    images_per_prompt = int(api_execution) // num_prompts
    remaining_images = int(api_execution) % num_prompts 

    image_count = 0
    for idx, prompt in enumerate(prompts):
        num_images = images_per_prompt + (1 if idx < remaining_images else 0)
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().replace(' ', '_') 
        clean_prompt = clean_prompt[:20]  

        for i in range(num_images):
            success = False
            attempts = 0
            files = {
                'prompt': (None, prompt),
                'width': (None, str(api_width)),
                'height': (None, str(api_height)),
                'steps': 5,
                'seed': (None, api_seed),
                'format': (None, api_format),
                'transparent_background': (None, str(api_transparent_background).lower())
            }

            while not success and attempts < max_retries:
                try:
                    response = requests.post(url, headers=headers, files=files)
                    response.raise_for_status() 

                    if response.status_code == 200:
                        image_count += 1
                        print(f'Successful request for prompt {idx+1}, image {image_count}')
                        image_filename = f'{clean_prompt}_{i+1}.jpg'
                        image_path = image_folder_path / image_filename
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                            print(f'Image saved at: {image_path}')
                        success = True
                    else:
                        print(f'Error in the request for image {image_count}: {response.status_code}, {response.text}')
                        attempts += 1
                        if attempts < max_retries:
                            print(f'Retrying in {retry_delay} seconds...')
                            time.sleep(retry_delay)
                except requests.exceptions.RequestException as e:
                    print(f'Request error: {e}')
                    attempts += 1
                    if attempts < max_retries:
                        print(f'Retrying in {retry_delay} seconds...')
                        time.sleep(retry_delay)

            if not success:
                print(f'Could not complete the request for image {image_count} after {max_retries} attempts.')


def text_to_speech_file(text: str, output_path: Path) -> str:
    try:
        response = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        with open(output_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"{output_path}: ¡Archivo de audio generado y guardado correctamente!")
    except Exception as e:
        print(f"Error al generar el audio: {e}")

    return output_path

def transcribe_audio_with_vosk(audio_file, model_path, transcription_file):
    model = Model(str(model_path))

    with wave.open(str(audio_file), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio debe estar en formato WAV mono PCM de 16 bits")
        
        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)

        words_data = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                for word in result.get("result", []):
                    words_data.append({
                        "word": word["word"],
                        "start_time": word["start"],
                        "end_time": word["end"]
                    })

        final_result = json.loads(recognizer.FinalResult())
        for word in final_result.get("result", []):
            words_data.append({
                "word": word["word"],
                "start_time": word["start"],
                "end_time": word["end"]
            })

    # Agrupar palabras en frases de hasta 8 palabras
    sentences = []
    current_sentence = []
    start_time = None

    for i, word_info in enumerate(words_data):
        if start_time is None:
            start_time = word_info["start_time"]

        current_sentence.append(word_info["word"])
        
        # Crear una frase en el JSON cada 8 palabras o al llegar al final
        if len(current_sentence) == 8 or i == len(words_data) - 1:
            end_time = word_info["end_time"]
            sentences.append({
                "text": " ".join(current_sentence),
                "start_time": start_time,
                "end_time": end_time
            })
            current_sentence = []
            start_time = None

    # Guardar la transcripción en el archivo JSON
    transcription_path = transcription_file
    try:
        with transcription_path.open(mode="w") as json_file:
            json.dump(sentences, json_file, indent=2)
        print(f"Transcripción guardada en {transcription_path}")
    except Exception as e:
        print(f"Error al guardar la transcripción: {e}")

    return transcription_path

def create_video_with_ffmpeg(image_folder, voiceover_file, background_file, transcription_file, specific_video_file, specific_audio_folder, fps, resolution, aspect_ratio, video_bitrate, font_name, font_color, font_size, font_border_color, encoder, pix_fmt, preset):
    try:
        if not specific_audio_folder.exists():
            specific_audio_folder.mkdir(parents=True)

        print("Combining background audio and voiceover...")
        voiceover_audio = AudioSegment.from_mp3(voiceover_file)
        background_audio = AudioSegment.from_mp3(background_file)
        
        background_audio = background_audio - 15 
        fade_duration = 1000
        background_audio = background_audio.fade_in(fade_duration).fade_out(fade_duration)

        total_audio_duration = len(voiceover_audio) + 5000
        if len(background_audio) < total_audio_duration:
            repeat_count = (total_audio_duration // len(background_audio)) + 1
            background_audio = background_audio * repeat_count

        background_audio = background_audio[:total_audio_duration]

        combined_audio = background_audio.overlay(voiceover_audio, position=2500)

        combined_audio_path = specific_audio_folder / "combined_audio.mp3"
        combined_audio.export(combined_audio_path, format="mp3")

        combined_audio_duration = len(combined_audio) / 1000
        print(f"Combined audio duration: {combined_audio_duration} seconds")

        images = [img for img in sorted(image_folder.iterdir()) if img.suffix in [".png", ".jpg", ".jpeg"]]
        if not images:
            print("No se encontraron imágenes en la carpeta proporcionada.")
            return

        num_images = len(images)
        duration_per_image = combined_audio_duration / num_images

        print("Creando video a partir de las imágenes...")
        filter_complex = ""
        for idx, image in enumerate(images):
            filter_complex += f"[{idx}:v]trim=duration={duration_per_image},setpts=PTS-STARTPTS[v{idx}];"

        filter_complex += "".join([f"[v{idx}]" for idx in range(num_images)]) + f"concat=n={num_images}:v=1[outv]"

        input_images = []
        for img in images:
            input_images.extend(["-loop", "1", "-t", str(duration_per_image), "-i", str(img)])

        ffmpeg_command_images = [
            "ffmpeg",
            *input_images,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-t", str(combined_audio_duration),
            "-c:v", str(encoder),  
            "-preset", str(preset), 
            "-pix_fmt", str(pix_fmt), 
            "-b:v", str(video_bitrate), 
            "-r", str(fps), 
            "-s", str(resolution),
            "-threads", str(cores),
            "-aspect", aspect_ratio,
            "temp_video.mp4"
        ]

        subprocess.run(ffmpeg_command_images, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        with transcription_file.open(mode="r") as f:
            transcription_data = json.load(f)

        text_delay = 2.5
        temp_input = "temp_video.mp4"

        for i, sentence_info in enumerate(transcription_data):
            text = sentence_info["text"]
            start_time = f"{float(sentence_info['start_time']) + text_delay:.2f}"
            end_time = f"{float(sentence_info['end_time']) + text_delay:.2f}"

            words = text.split()
            if len(words) > 4:
                line1 = " ".join(words[:4])  # Primeras 4 palabras
                line2 = " ".join(words[4:])  # Resto de las palabras
            else:
                line1 = text
                line2 = ""

            drawtext_command_line1 = (
                f"drawtext=fontfile='{font_name}':text='{line1}':"
                f"fontcolor={font_color}:fontsize={font_size}:x=(w-text_w)/2:y=(h-text_h)/2-40:borderw=2:bordercolor={font_border_color}:"
                f"shadowx=2:shadowy=2:shadowcolor=black:enable='between(t\\,{start_time}\\,{end_time})'"
            )

            drawtext_command_line2 = (
                f"drawtext=fontfile='{font_name}':text='{line2}':"
                f"fontcolor={font_color}:fontsize={font_size}:x=(w-text_w)/2:y=(h-text_h)/2+40:borderw=2:bordercolor={font_border_color}:"
                f"shadowx=2:shadowy=2:shadowcolor=black:enable='between(t\\,{start_time}\\,{end_time})'"
            ) if line2 else ""

            filter_complex_text = drawtext_command_line1
            if line2:
                filter_complex_text += f",{drawtext_command_line2}"

            output_file = f"temp_video_text_{i}.mp4"
            
            ffmpeg_command = [
                "ffmpeg", "-i", temp_input, "-c:v", str(encoder), "-preset", str(preset), "-pix_fmt", str(pix_fmt), 
                "-b:v", str(video_bitrate), "-r", str(fps), "-s", str(resolution), "-threads", str(cores), 
                "-aspect", aspect_ratio, "-vf", filter_complex_text, "-c:a", "copy", str(output_file)
            ]

            subprocess.run(ffmpeg_command, shell=True, check=True)
            
            if temp_input != "temp_video.mp4":
                Path(temp_input).unlink(missing_ok=True)

            temp_input = output_file

        final_ffmpeg_command = [
            "ffmpeg",
            "-i", temp_input, 
            "-i", str(combined_audio_path),  
            "-c:v", str(encoder), 
            "-preset", str(preset), 
            "-pix_fmt", str(pix_fmt), 
            "-b:v", str(video_bitrate),  
            "-r", str(fps), 
            "-s", str(resolution),
            "-threads", str(cores),  
            "-aspect", aspect_ratio,
            "-c:a", "aac", 
            "-b:a", "192k", 
            "-shortest",
            str(specific_video_file)  
        ]
        
        subprocess.run(final_ffmpeg_command, shell=True, check=True)

        # Eliminar el archivo temporal final si existe
        if Path(temp_input).exists():
            Path(temp_input).unlink(missing_ok=True)
        
        # Eliminar los archivos temporales iniciales
        if Path("temp_video.mp4").exists():
            Path("temp_video.mp4").unlink(missing_ok=True)
        
        print(f"Video generado con superposición de texto guardado como {specific_video_file}")

    except Exception as e:
        print(f"Error al crear el video: {e}")


def process_videos_from_excel(excel_path):
    
    df = pd.read_excel(excel_path)

    for index, row in df.iterrows():
        prompt = row['prompt_dezgo']
        audio_prompt = row['voiceover_prompt']
        num_images = row['num_images']
        background_audio = row['background_audio']
        output_name = row['name_video']  
        fps =  row['fps'] 
        resolution = row['resolution']
        aspect_ratio = row['aspect_ratio']
        video_bitrate = row['video_bitrate']
        encoder = row['encoder']
        quality_level = row['quality_level']
        font_name = row['font_name']
        font_color = row['font_color']
        font_size = row['font_size']
        font_border_color = row['font_border_color']

        specific_image_folder = image_folder / f"video_{index}"
        specific_audio_folder = audio_folder / f"audio_{index}"
        specific_audio_file = specific_audio_folder / f"audio_{index}.mp3"
        specific_video_file = video_folder / f"{output_name}.mp4" 
        specific_transcription_file = transcription_folder / f"transcription_{index}.json"

        if not specific_image_folder.exists():
            specific_image_folder.mkdir()
        if not specific_audio_folder.exists():
            specific_audio_folder.mkdir()

        quality_presets = {
            1: 'veryslow',  # Best quality
            2: 'medium',    # Medium quality
            3: 'veryfast'   # Low quality
        }

        encoder_settings = {
            'libx264': {'pix_fmt': 'yuv420p','presets': quality_presets},
            'h264_nvenc': {'pix_fmt': 'yuv420p','presets': {1: 'p1', 2: 'p4', 3: 'p7'}},
            'h264_qsv': {'pix_fmt': 'nv12','presets': quality_presets},
            'h264_amf': {'pix_fmt': 'nv12','presets': quality_presets},
            'h264_videotoolbox': {'pix_fmt': 'nv12','presets': {1: 'slow', 2: 'medium', 3: 'fast'}},
            'hevc_nvenc': {'pix_fmt': 'yuv420p','presets': {1: 'p1', 2: 'p4', 3: 'p7'}},
            'hevc_qsv': {'pix_fmt': 'nv12','presets': quality_presets},
            'hevc_amf': {'pix_fmt': 'nv12','presets': quality_presets},
            'hevc_videotoolbox': {'pix_fmt': 'nv12','presets': {1: 'slow', 2: 'medium', 3: 'fast'}},
            'vp9_nvenc': {'pix_fmt': 'yuv420p','presets': {1: 'p1', 2: 'p4', 3: 'p7'}},
            'vp9_qsv': {'pix_fmt': 'nv12','presets': quality_presets},
            'vp9_amf': {'pix_fmt': 'nv12','presets': quality_presets},
            'vp9_videotoolbox': {'pix_fmt': 'nv12','presets': {1: 'slow', 2: 'medium', 3: 'fast'}},
            'av1_nvenc': {'pix_fmt': 'yuv420p','presets': {1: 'p1', 2: 'p4', 3: 'p7'}},
            'av1_qsv': {'pix_fmt': 'nv12','presets': quality_presets},
            'av1_amf': {'pix_fmt': 'nv12','presets': quality_presets},
            'av1_videotoolbox': {'pix_fmt': 'nv12','presets': {1: 'slow', 2: 'medium', 3: 'fast'}},
            'libx265': {'pix_fmt': 'yuv420p','presets': {1: 'veryslow', 2: 'medium', 3: 'fast'}}
        }
        encoder_name = encoder
        encoder_settings = encoder_settings[encoder_name]  # Get the settings for this encoder
        preset = encoder_settings['presets'][quality_level]
        pix_fmt = encoder_settings['pix_fmt']
        encoder = encoder_name


        #create_images_ia(api_key, url_api, api_endpoint, prompt, api_width, api_height, api_sampler, api_model_id, api_negative_prompt, api_seed, api_format, api_guidance, api_transparent_background, num_images, specific_image_folder)
        
        text_to_speech_file(audio_prompt, specific_audio_file)

        wav_file = convert_audio_to_wav(specific_audio_file, specific_audio_folder)

        transcribe_audio_with_vosk(wav_file, model_path, specific_transcription_file)

        create_video_with_ffmpeg(specific_image_folder, specific_audio_file, audio_folder / background_audio, specific_transcription_file, specific_video_file, specific_audio_folder, fps, resolution, aspect_ratio, video_bitrate, font_name, font_color, font_size, font_border_color, encoder, pix_fmt, preset)

        print(f"Video generado y guardado como {specific_video_file}")


if __name__ == "__main__":
    setup_fontconfig(base_path)

    excel_path = base_path / "video_data.xlsx"

    process_videos_from_excel(excel_path)
