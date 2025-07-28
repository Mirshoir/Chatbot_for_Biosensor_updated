from gradio_client import Client

def analyze_image_with_gradio(image_path, prompt="""You are a Vision-Language Model assistant.
Your task is to detect any visual signs that might affect the user's cognitive load.
Analyze the image for:
- Distractions in the background (other screens, phone, clutter)
- User’s posture (slouching, leaning, alert)
- Facial signs of strain, confusion, or engagement
- Visible multi-tasking (holding phone, other apps open)

Output a short summary:
- Environment distractions: [describe]
- User posture: [describe]
- Facial cues: [describe]
- Any other factor that might impact focus.
"""):
    try:
        print("Initializing primary Gradio client (ybelkada/llava-1.5-dlai)...")
        client = Client("ybelkada/llava-1.5-dlai")

        with open(image_path, "rb") as img_file:
            result = client.predict(
                prompt,       # text input
                img_file,     # binary file object (image)
                api_name="/predict"
            )

        print("Received response from primary client.")
        return result

    except Exception as e:
        print(f"Primary client error: {e}")
        print("Switching to fallback client (Mirshoir/llava-1.5-dlai)...")

        try:
            client = Client("Mirshoir/llava-1.5-dlai")

            with open(image_path, "rb") as img_file:
                result = client.predict(
                    prompt,
                    img_file,
                    api_name="/predict"
                )

            print("Received response from fallback client.")
            return result

        except Exception as e2:
            print(f"Fallback client error: {e2}")
            return {"error": f"Both clients failed. Errors: {e} | {e2}"}
