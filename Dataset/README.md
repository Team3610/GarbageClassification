#  쓰레기 분류 학습 데이터셋 (Garbage Classification Dataset)

본 폴더(`Dataset/`)는 모델 학습 및 검증에 사용되는 원본 이미지 데이터셋을 보관하는 위치입니다. 
데이터셋 전체 용량이 큰 관계로, 파일들은 GitHub 저장소에 직접 업로드하지 않고 클라우드 스토리지를 통해 공유합니다. (현재 `.gitignore`에 의해 이미지 파일들은 추적되지 않습니다.)

##  데이터셋 다운로드 및 세팅 방법

1. 아래의 구글 드라이브 링크에 접속하여 `dataset.zip` 파일을 다운로드합니다.
   * **다운로드 링크:** [https://drive.google.com/file/d/1L8TpC9F72u0hcoA-kqcD3gvvgZNQvrPn/view?usp=sharing]
2. 다운로드한 압축 파일을 이 `Dataset` 폴더 내부에 압축 해제합니다.
3. 압축 해제 후, 폴더 구조가 아래와 같이 구성되었는지 확인해 주세요.

## 📂 폴더 구조 안내

```text
Dataset/
├── .gitkeep
├── README.md
├── battery/            
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/