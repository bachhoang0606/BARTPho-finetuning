import os
import sys

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils import handle_raw_datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('Bachhoang/gpt2-vietnamese-legal')
model = GPT2LMHeadModel.from_pretrained('Bachhoang/gpt2-vietnamese-legal')
"""
##### Câu hỏi: Nội dung của Điều 11 27/2022/NQ-HĐND là gì? ### Trả lời: 
Điều 11 27/2022/NQ-HĐND về định mức phân bổ dự toán chi thường xuyên ngân sách địa phương năm 2023 
và giai đoạn 2023-2025 trên địa bàn tỉnh khánh hòa c) Kinh phí tăng hệ số lương theo định kỳ, tăng 
lương trước thời hạn. d) Kinh phí vận hành, duy trì và hoạt động cho trang/cổng thông tin điện tử 
của các cơ quan, đơn vị, địa phương và các xã, phường, thị trấn (đối với chế độ thù lao, nhuận bút 
được phân bổ thêm ngoài định mức). đ) Kính phí sửa chữa tài sản phục vụ công tác chuyên môn và bảo 
dưỡng thường xuyên các công trình cơ sở hạ tầng, ô tô quy mô nhỏ; kinh phí mua sắm, thay thế máy móc, 
thiết bị văn phòng phổ biến cho cán bộ, công chức và máy móc, thiết bị văn phòng phổ biến (trừ máy photocopy, 
máy lạnh) phục vụ công tác hành chính, văn thư, tiếp dân, bộ phận một cửa của cơ quan, tổ chức, đơn vị theo 
quy định tại Quyết định số 50/2017/QĐ-TTg ngày 31 tháng 12 năm 2017 của Thủ tướng Chính phủ quy định tiêu chuẩn, 
định mức sử dụng máy móc, thiết bị (đối với trường hợp Lãnh đạo cơ quan, đơn vị nhận công tác mới được phân bổ 
thêm ngoài định mức). 2. Định mức phân bổ không bao gồm chi tiền lương, các khoản có tính chất tiền lương và các 
khoản đóng góp theo lương (đối với biên chế được giao nhưng chưa tuyển được xác định trên cơ sở hệ số lương bậc 1 
của ngạch vị trí tuyển dụng chuyên viên là 2,34). 3. Đối với cấp tỉnh a) Các cơ quan quản lý nhà nước - Tiêu chí bổ 
sung cơ quan tổng hợp thực hiện các nhiệm vụ phát sinh do cơ quan có thẩm quyền giao: Sở Kế hoạch và Đầu tư, Sở Tài 
chính: 200 triệu đồng/đơn vị/năm. - Văn phòng Ủy ban nhân dân tỉnh; Đoàn Đại biểu quốc hội và Hội đồng nhân dân tỉnh
 được bố trí dự toán ngoài định mức chung theo khả năng cân đối ngân sách để thực hiện một số nhiệm vụ phát sinh do 
 cơ quan có thẩm quyền giao. b) Đối với cơ quan Đảng Tiêu chí bổ sung: Văn phòng Tỉnh ủy được bố trí dự toán ngoài định
   mức chung theo khả năng cân đối ngân sách để thực hiện một số nhiệm vụ phát sinh do cơ quan có thẩm quyền giao. #####

"""
max_length = 512

def inference_gpt(text : str):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id,
                                   do_sample=True,
                                   max_length=max_length,
                                   min_length=max_length,
                                   no_repeat_ngram_size=2,
                                   num_return_sequences=3,
                                   )
    for i, sample_output in enumerate(sample_outputs):
        raw_ouput = tokenizer.decode(sample_output.tolist())
        return handle_raw_datasets(raw_ouput, text)

if __name__ == "__main__":

    text = "##### Câu hỏi: Nội dung của Điều 11 27/2022/NQ-HĐND là gì? ### Trả lời:"
    input_ids = tokenizer.encode(text, return_tensors='pt')
    sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id,
                                   do_sample=True,
                                   max_length=max_length,
                                   min_length=max_length,
                                   no_repeat_ngram_size=2,
                                   num_return_sequences=3,
                                   )
    for i, sample_output in enumerate(sample_outputs):
        raw_ouput = tokenizer.decode(sample_output.tolist())
        print(">> Generated text {}\n\n{}".format(i+1, handle_raw_datasets(raw_ouput, text).strip()))
        print('\n---')
