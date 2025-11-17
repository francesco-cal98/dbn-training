# ============================================================================
# 🌐 INTERACTIVE PCA EXPLORER - Streamlit App
# ============================================================================
# Aggiungi questa cella alla fine del notebook cc_lab_01.ipynb
# dopo aver addestrato il DBN (dopo la cella del training)
# ============================================================================

import os

# Detect environment
IN_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())

if IN_COLAB:
    print("="*70)
    print("🚀 Launching Interactive PCA Explorer on Google Colab")
    print("="*70)
    print()

    # Install dependencies (quiet mode)
    print("📦 Installing streamlit and plotly...")
    get_ipython().system('pip install -q streamlit plotly')
    print("✅ Dependencies installed")
    print()

    # Save the trained model
    print("💾 Saving trained DBN model...")
    model_path = '/content/groundeep-unimodal-training/dbn_mnist.pkl'
    dbn_mnist.save(model_path)
    print()

    # Launch Streamlit in background
    print("🌐 Starting Streamlit server (port 8501)...")
    get_ipython().system('streamlit run /content/groundeep-unimodal-training/pca_explorer_colab.py &>/content/logs.txt &')

    # Wait for server to start
    import time
    time.sleep(3)
    print("✅ Streamlit server started")
    print()

    # Create public tunnel
    print("📡 Creating public URL with LocalTunnel...")
    print("⏳ This may take 10-20 seconds...")
    print()

    # Run localtunnel and show the URL
    get_ipython().system('npx localtunnel --port 8501')

    print()
    print("="*70)
    print("✅ APP IS RUNNING!")
    print("="*70)
    print()
    print("🔗 CLICK THE URL ABOVE (looks like: https://xxx-xxx.loca.lt)")
    print()
    print("📝 If you see a 'Tunnel Password' page:")
    print("   → Click the 'Click to Continue' button")
    print("   → Or enter the IP address shown above")
    print()
    print("⚠️  IMPORTANT: Keep this cell running!")
    print("   Stopping it will close the app.")
    print()
    print("💡 To check logs if something goes wrong:")
    print("   !cat /content/logs.txt")
    print("="*70)

else:
    # Running locally
    print("="*70)
    print("💻 Not on Colab - Running Locally")
    print("="*70)
    print()
    print("To launch the app locally, run in terminal:")
    print()
    print("  streamlit run pca_explorer.py")
    print()
    print("The app will open at http://localhost:8501")
    print("="*70)
