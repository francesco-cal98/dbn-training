"""
Quick test script to verify DBN and RBM wrapper functionality.

This tests the core API methods that are used in cc_lab_01 and cc_lab_02.
"""

import torch
import sys

print("="*60)
print("Testing DBN and RBM Wrappers")
print("="*60)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from DBN import DBN
    from RBM import RBM
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create DBN instance (matching cc_lab_01 parameters)
print("\n[Test 2] Creating DBN instance...")
try:
    dbn = DBN(
        visible_units=28*28,          # MNIST flattened
        hidden_units=[400, 500, 800], # Architecture from cc_lab_01
        k=1,
        learning_rate=0.1,
        learning_rate_decay=False,
        initial_momentum=0.5,
        final_momentum=0.95,
        weight_decay=0.0001,
        xavier_init=False,
        increase_to_cd_k=False,
        use_gpu=torch.cuda.is_available()
    )
    print("✅ DBN created successfully")
    print(f"   Architecture: {dbn.layer_sizes}")
    print(f"   Number of layers: {len(dbn.rbm_layers)}")
except Exception as e:
    print(f"❌ DBN creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check rbm_layers attribute (critical for cc_lab_01)
print("\n[Test 3] Checking rbm_layers attribute...")
try:
    assert len(dbn.rbm_layers) == 3, f"Expected 3 layers, got {len(dbn.rbm_layers)}"
    print(f"✅ rbm_layers has {len(dbn.rbm_layers)} layers")

    # Check each layer
    for i, rbm in enumerate(dbn.rbm_layers):
        print(f"   Layer {i}: {rbm.visible_units} -> {rbm.hidden_units}")
except Exception as e:
    print(f"❌ rbm_layers check failed: {e}")
    sys.exit(1)

# Test 4: Check W attribute access (used in cc_lab_01 for weight visualization)
print("\n[Test 4] Checking weight matrix access...")
try:
    w1 = dbn.rbm_layers[0].W
    w2 = dbn.rbm_layers[1].W
    w3 = dbn.rbm_layers[2].W

    assert w1.shape == (784, 400), f"Expected (784, 400), got {w1.shape}"
    assert w2.shape == (400, 500), f"Expected (400, 500), got {w2.shape}"
    assert w3.shape == (500, 800), f"Expected (500, 800), got {w3.shape}"

    print("✅ Weight matrices accessible and correct shape")
    print(f"   W1: {w1.shape}")
    print(f"   W2: {w2.shape}")
    print(f"   W3: {w3.shape}")
except Exception as e:
    print(f"❌ Weight access failed: {e}")
    sys.exit(1)

# Test 5: Test to_hidden method (critical for cc_lab_01)
print("\n[Test 5] Testing to_hidden() method...")
try:
    # Create dummy input (10 samples of MNIST)
    dummy_input = torch.randn(10, 784)

    # Test to_hidden on first layer
    hidden_repr, hidden_sample = dbn.rbm_layers[0].to_hidden(dummy_input)

    assert hidden_repr.shape == (10, 400), f"Expected (10, 400), got {hidden_repr.shape}"
    assert hidden_sample.shape == (10, 400), f"Expected (10, 400), got {hidden_sample.shape}"

    print("✅ to_hidden() works correctly")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Hidden repr shape: {hidden_repr.shape}")
    print(f"   Hidden sample shape: {hidden_sample.shape}")
except Exception as e:
    print(f"❌ to_hidden() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test forward pass through DBN
print("\n[Test 6] Testing full forward pass...")
try:
    dummy_input = torch.randn(10, 784)
    h_prob, h_sample = dbn.forward(dummy_input)

    assert h_prob.shape == (10, 800), f"Expected (10, 800), got {h_prob.shape}"

    print("✅ Forward pass works correctly")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {h_prob.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test reconstruction
print("\n[Test 7] Testing reconstruction...")
try:
    dummy_input = torch.randn(10, 784)
    reconstruction = dbn.reconstruct(dummy_input)

    assert reconstruction.shape == (10, 784), f"Expected (10, 784), got {reconstruction.shape}"

    print("✅ Reconstruction works correctly")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Reconstruction: {reconstruction.shape}")
except Exception as e:
    print(f"❌ Reconstruction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test mini training (very short to verify API)
print("\n[Test 8] Testing training API (1 epoch, small data)...")
try:
    # Create tiny fake MNIST data
    fake_data = torch.randn(50, 784)  # 50 samples
    fake_labels = torch.randint(0, 10, (50,))

    # Train for just 1 epoch to test the API
    dbn.train_static(
        fake_data,
        fake_labels,
        num_epochs=1,
        batch_size=10
    )

    print("✅ Training API works correctly")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Verify weights changed after training
print("\n[Test 9] Verifying weights changed after training...")
try:
    # Get weight before training
    w_before = dbn.rbm_layers[0].W.clone()

    # Train for 1 more epoch
    fake_data = torch.randn(50, 784)
    fake_labels = torch.randint(0, 10, (50,))
    dbn.train_static(fake_data, fake_labels, num_epochs=1, batch_size=10)

    # Get weight after training
    w_after = dbn.rbm_layers[0].W

    # Check if they're different (training should have updated them)
    assert not torch.allclose(w_before, w_after), "Weights didn't change after training!"

    print("✅ Weights updated correctly during training")
except Exception as e:
    print(f"❌ Weight update verification failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("Wrappers are ready for use in cc_lab notebooks")
print("="*60)
