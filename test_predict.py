"""
test_predict.py — Quick sanity check before testing through chatbot.py
========================================================================
Run this first to confirm:
  1. skin_model.pth loads correctly
  2. predict_image() returns sensible output
  3. MedicalChatbot.analyze_image() wiring works

Usage:
    python test_predict.py path/to/test_image.jpg
"""

import sys
from pathlib import Path


def test_predict_skin_direct(image_path: str):
    print("=" * 60)
    print("TEST 1: predict_skin.predict_image() directly")
    print("=" * 60)
    from predict_skin import predict_image

    results = predict_image(image_path, top_k=3)
    print(f"Image: {image_path}")
    for label, conf in results:
        print(f"  {label:60s} {conf*100:5.2f}%")
    print()
    return results


def test_predict_skin_tta(image_path: str):
    print("=" * 60)
    print("TEST 2: predict_skin.predict_image() with TTA")
    print("=" * 60)
    from predict_skin import predict_image

    results = predict_image(image_path, top_k=3, tta=True, tta_n=5)
    for label, conf in results:
        print(f"  {label:60s} {conf*100:5.2f}%")
    print()
    return results


def test_chatbot_analyze_image(image_path: str):
    print("=" * 60)
    print("TEST 3: MedicalChatbot.analyze_image() wiring")
    print("=" * 60)
    from chatbot import MedicalChatbot

    # Note: this will also load SentenceTransformer + faiss index,
    # so it's slower than test 1/2 — but confirms the real code path.
    assistant = MedicalChatbot()
    results = assistant.analyze_image(image_path)
    formatted = assistant._format_image_results(results)
    for item in formatted:
        print(f"  {item['label']:60s} {item['confidence']*100:5.2f}%")
    print()
    return formatted


def test_chatbot_full_answer(image_path: str):
    print("=" * 60)
    print("TEST 4: Full MedicalChatbot.answer() with image only")
    print("=" * 60)
    from chatbot import MedicalChatbot

    assistant = MedicalChatbot()

    with open(image_path, "rb") as f:
        import base64
        encoded = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded}"

    result = assistant.answer(
        query="",
        mode="dermatology",
        image_data_url=data_url,
        image_file_name=Path(image_path).name,
        image_mime_type="image/jpeg",
    )

    print("Reply:", result["reply"])
    print("Source:", result["source"])
    print("Image predictions:", result.get("image_predictions"))
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    # Run tests incrementally — comment out later ones if earlier ones fail
    test_predict_skin_direct(image_path)
    test_predict_skin_tta(image_path)
    test_chatbot_analyze_image(image_path)
    test_chatbot_full_answer(image_path)

    print("All tests completed.")