from pyngrok import ngrok
# Setup and start ngrok tunnel
try:
    ngrok.kill()
except:
    pass

ngrok.set_auth_token("2vmuVoIZNe5aF3F9ZjaRAM4LXCw_5cW11S3Y362yNKJfUBB8B")
public_url = ngrok.connect(8055)
print("Public URL:", public_url)
