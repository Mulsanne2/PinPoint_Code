#!/bin/bash

# 프로세스 이름을 입력으로 받아서 종료하는 함수
kill_processes_by_name() {
    process_name=$1

    if [ -z "$process_name" ]; then
        echo "Usage: $0 <process_name>"
        exit 1
    fi

    # 주어진 이름의 프로세스를 찾아서 해당하는 프로세스 ID를 가져옵니다.
    processes=$(ps -ef | grep "$process_name" | grep -v "grep" | awk '{print $2}')

    # 가져온 각각의 프로세스 ID에 대해 종료 시도
    for pid in $processes; do
        echo "Killing process $pid"
        kill $pid
    done
}

# 사용 예시: ./script_name.sh process_name
# 명령행 인자를 받아서 실행
kill_processes_by_name "$1"