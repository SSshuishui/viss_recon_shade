#!/bin/bash

# 遍历文件名的数字范围
# for j in {1..130}
# do
#     # 构建文件名
#     file="frequency_10M/uvw${j}frequency10M.txt"

#     # 检查文件是否存在
#     if [ -f "$file" ]; then
#         # 统计文件的行数并打印
#         lines=$(wc -l < "$file")
#         echo "$file lines: $lines"
#     else
#         # 如果文件不存在，则提示并跳过
#         echo "$file does not exist. Skipping..."
#     fi
# done


# # 遍历文件名的数字范围
# for j in {1..130}
# do
#     # 构建文件名
#     file="frequency_30M/updated_uvw${j}frequency30M.txt"

#     # 检查文件是否存在
#     if [ -f "$file" ]; then
#         # 统计文件的行数并打印
#         lines=$(wc -l < "$file")
#         echo "$file lines: $lines"
#     else
#         # 如果文件不存在，则提示并跳过
#         echo "$file does not exist. Skipping..."
#     fi
# done


# 初始化行数总和变量

for j in {1..130}
do
    # 构建文件名
    file="frequency_1M/shadeM${j}frequency1M.txt"

    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 统计文件的行数并累加到总行数
        lines=$(wc -l < "$file")
        echo "$file lines: $lines"
    else
        # 如果文件不存在，则提示并跳过
        echo "$file does not exist. Skipping..."
    fi
done



# 初始化最大行数变量
# max_lines=0

# for j in {1..130}
# do
#     # 构建文件名
#     file="frequency_30M/updated_uvw${j}frequency30M.txt"

#     # 检查文件是否存在
#     if [ -f "$file" ]; then
#         # 统计文件的行数
#         lines=$(wc -l < "$file")
        
#         # 打印文件的行数
#         echo "$file lines: $lines"
        
#         # 如果当前行数大于已知的最大行数，则更新最大行数
#         if [ "$lines" -gt "$max_lines" ]; then
#             max_lines=$lines
#         fi
#     else
#         # 如果文件不存在，则提示并跳过
#         echo "$file does not exist. Skipping..."
#     fi
# done

# # 打印最大的行数
# echo "Maximum lines: $max_lines"