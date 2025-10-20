import csv

# 输入文件路径（请替换为你的实际文件路径）
input_file = "C:\\Users\\张喻飞\\PycharmProjects\\NUS-E-Commerce-Sentiment-Analysis-Platform\\data\\train.csv"
# 输出文件路径（提取第一列后的新文件）
output_file = "extracted_sentences.csv"

# 读取原始CSV并提取第一列
with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    # 创建CSV读取器和写入器
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 读取表头并写入新文件（保留第一列的表头"sentence"）
    header = next(reader)
    writer.writerow([header[0]])  # 只写入第一列的表头

    # 遍历每行，提取第一列并写入新文件
    for row in reader:
        if row:  # 跳过空行
            writer.writerow([row[0]])  # 写入当前行的第一列

print(f"已成功提取第一列到 {output_file}")