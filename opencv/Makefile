# Makefile

# 컴파일러
CXX = g++

# 컴파일 플래그 (C++17 사용 가능하게)
CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`

# 링크 플래그 (OpenCV 라이브러리)
LDFLAGS = `pkg-config --libs opencv4`

# 타겟 이름
TARGET = palm

# 소스 파일
SRCS = gesture_cpp.cpp

# 기본 빌드 규칙
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# clean 명령: 생성된 파일 삭제
clean:
	rm -f $(TARGET)
