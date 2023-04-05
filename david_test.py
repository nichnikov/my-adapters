import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer
model_name = 'cointegrated/rubert-base-cased-dp-paraphrase-detection'
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
tokenizer = BertTokenizer.from_pretrained(model_name)

def compare_texts(text1, text2):
    batch = tokenizer(text1, text2, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        proba = torch.softmax(model(**batch).logits, -1).cpu().numpy()
    return proba[0] # p(non-paraphrase), p(paraphrase)

print(compare_texts('нужно ли подавать уведомление об исчисленных суммах по патенту?',
                    'как подать уведомление по усн об исчисленных суммах'))
# [0.7056226 0.2943774]
print(compare_texts('каким образом подавать уведомление об исчисленных суммах по усн?',
                    'как подать уведомление по усн об исчисленных суммах'))
# [0.16524374 0.8347562 ]
print(compare_texts('каким образом подавать уведомление об исчисленных суммах по ндс?',
                    'как подать уведомление по усн об исчисленных суммах'))
