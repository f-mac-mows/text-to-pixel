import os
import sys
from pixel_deserializer import PixelDeserializer

def main():
    # 💡 os.path.expanduser를 사용해야 '~' 기호가 /Users/유저명/ 형태의 실제 절대 경로로 치환됩니다.
    target_image_path = os.path.expanduser("~/Assets/Sprites/test.png")
    
    print("=" * 60)
    print(f"🛰️  PIXEL ART REVERSE PIPELINE (DESERIALIZER)")
    print(f"📂 타깃 이미지: {target_image_path}")
    print("=" * 60)
    
    # 1. 파일이 진짜 그 자리에 존재하는지 방어 코드 가동
    if not os.path.exists(target_image_path):
        print(f"[❌ 에러] 지정한 경로에 test.png 파일이 없습니다.")
        print(f"⚠️  현재 경로 확인을 부탁드립니다!")
        sys.exit(1)
        
    try:
        # 2. 역직렬화 가동 및 색상 매핑 연산
        print("[*] 📦 16x16 픽셀 디코딩 및 가중치 역산 중...")
        protocol_result = PixelDeserializer.deserialize_image(target_image_path)
        
        # 3. 최종 압축 프로토콜 화면 출력
        print("\n✨ [SUCCESS] 역직렬화 파싱 성공!")
        print("-" * 60)
        print(f"🔮 Output Protocol:\n{protocol_result}")
        print("-" * 60)
        
    except Exception as e:
        print(f"[❌ 에러] 역직렬화 중 알 수 없는 예외 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()