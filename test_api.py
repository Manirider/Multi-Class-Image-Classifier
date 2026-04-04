import time
import requests
import sys
BASE_URL = "http://127.0.0.1:8000"
def test_api():
    print("Waiting for API to boot...")
    for _ in range(15):
        try:
            res = requests.get(f"{BASE_URL}/health")
            if res.status_code == 200:
                print("API is UP!")
                break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("API failed to boot.")
        sys.exit(1)
    print("\n--- Test 1: Empty Predict Request ---")
    res = requests.post(f"{BASE_URL}/predict") 
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 400
    print("\n--- Test 2: Invalid File Upload ---")
    files = {'file': ('dummy.txt', b"not an image", 'text/plain')}
    res = requests.post(f"{BASE_URL}/predict", files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 422
    print("\n--- Test 3: Corrupt Image Upload ---")
    files = {'file': ('corrupt.jpg', b"corrupt image bytes but right mimetype", 'image/jpeg')}
    res = requests.post(f"{BASE_URL}/predict", files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 422
    print("\n--- Test 4: Valid Image Upload ---")
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    res = requests.post(f"{BASE_URL}/predict", files=files)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert 'predicted_class' in res.json()
    assert 'confidence' in res.json()
    print("\nAll API tests PROVEN SUCCESSFUL.")
if __name__ == "__main__":
    test_api()
