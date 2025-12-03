## 🚀 3. 개발 환경 설정 (Development Setup)

본 프로젝트는 Python 3.x 기반의 **가상 환경(Virtual Environment)** 사용을 권장합니다.

### 3.1 가상 환경 생성 및 활성화

프로젝트 루트 디렉토리에서 다음 명령어를 순서대로 실행합니다.

1.  **가상 환경 생성:**
    ```bash
    python3 -m venv venv
    ```

2.  **가상 환경 활성화:**
    * **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate
        ```
    > 💡 **참고:** 활성화되면 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

### 3.2 의존성 설치

가상 환경이 활성화된 상태에서, `requirements.txt` 파일을 사용하여 필요한 모든 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 3.3 탐지 명령어
```bash
python3 detect.py --image assets/person.jpg --weight best_model.pth
```

### 3.4 실행 명령어
```bash
python3 main.py --epoch 50 --batch_size 48
```

### 3.5 평가 명령어
```bash
python3 eval.py --weight best_model.pth
```


### 3.6 skip-denet 실행 명령어
```bash
python3 main.py --epoch 50 --batch_size 48 --lr 0.0001
```
