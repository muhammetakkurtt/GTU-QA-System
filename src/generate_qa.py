# Loading text from file
with open("test.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Breaking text into paragraphs or meaningful sections
contexts = text.split("\n\n")  # We separate using double line spaces
print(f"Toplam {len(contexts)} bağlam bulundu.")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Installing the LLaMA 3.1-8B-Instruct model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",  # Automatically uses the GPU
    torch_dtype=torch.float16  # Optimizes GPU memory
)

# Pipeline creation
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate questions and answers from a context
def generate_questions_and_answers(context, num_questions=10):
    prompt = f"""
    Sen bir eğitim uzmanısın ve yönetmelik metinlerinden soru-cevap oluşturmakta uzmanlaşmışsın.
    Aşağıdaki bağlama göre TAM OLARAK {num_questions} TÜRKÇE SORU ve CEVAP oluştur.

    Kurallar:
    1. KESİNLİKLE TÜRKÇE KULLAN! İNGİLİZCE KULLANMANI İSTEMİYORUM!
    2. Sadece {num_questions} soru sor
    3. Sorular bağlamla doğrudan ilgili olmalı
    4. Her cevap metinden BİREBİR ALINTI olmalı
    5. Sorular ve cevaplar şu formatta oluşturulmalı:
       {{
          "soru": "[Soru metni]",
          "cevap": "[Metinden direkt alıntı] (Madde X-(Y))"
       }}
    6. Sorular numaralandırılmalı (1'den {num_questions}'a kadar)
    7. JSON formatında liste şeklinde oluşturulmalı.
    8. Sorular açık, anlaşılır ve detaylı olmalı
    9. Cevaplar mümkün olduğunca uzun ve açıklayıcı olmalı
    10. Yönetmeliğin önemli noktalarını vurgulayan sorular sorulmalı
    11. Her cevabın sonunda kaynak madde numarası eklenmelidir (örn: Madde 1-(1))
    12. Tüm cevaplar metinden birebir alıntıdır.
    
    Bağlam:
    {context}

    Yukarıdaki kurallara uygun şekilde JSON formatında {num_questions} soru ve cevap oluştur:
    """
    
    result = qa_pipeline(
        prompt, 
        max_new_tokens=2048,
        num_return_sequences=1,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2
    )
    return result[0]["generated_text"]

# We operate on the first few contexts (first 2 contexts as an example)
qa_results = []
for i, context in enumerate(contexts[:2]):  # First 2 context
    print(f"Bağlam {i+1} işleniyor...")
    qa_output = generate_questions_and_answers(context)
    qa_results.append({
        "context": context,
        "qa_output": qa_output
    })

# Function to save the results to a file
def save_qa_results(qa_results, output_file="qa_results.txt"):
    with open(output_file, "w", encoding="utf-8") as file:
        for i, result in enumerate(qa_results, 1):
            file.write(f"\n=== Bağlam {i} ===\n")
            file.write("\nBağlam:\n")
            file.write(result["context"])
            file.write("\n\nSorular ve Cevaplar:\n")
            file.write(result["qa_output"])
            file.write("\n" + "="*50 + "\n")


save_qa_results(qa_results)
