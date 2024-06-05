for dir in */; do
    if [ -d "$dir" ]; then
        if [ -f "${dir}opponent.py" ]; then
            # 如果存在，则先删除该文件
            rm "${dir}opponent.py"
        fi
        ln -s $(pwd)/opponent.py "${dir}opponent.py"
    fi
done