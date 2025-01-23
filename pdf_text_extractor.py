import pdfplumber

pdf_files = [
    "YÖ-0005_Dikey_Geçiş_Lisans_Uygulama_Yönergesi R1.pdf",
    "YÖ-0032_Prof._Dr._Nejat_Goyunc_Kutuphanesi_Hizmetlerinden_Yararlanma_Yonergesi R1.pdf",
    "YÖ-0004 Çift Anadal Programı Yönergesi R2.pdf",
    "YN-0001 Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği R7.pdf",
    "YN-0009_Yonetmelik_Formu_Lisans_Egitim_ve_Ogretim_Yonetmeligi-2.pdf",
    "YÖ-0002 Önlisans-Lisans İngilizce Hazırlık Eğitim-Öğretim ve Sınav Yönergesi R7.pdf",
    "YÖ-0012 Yandal Programı Yönergesi R3.pdf", 
    "YÖ-0013 Önlisans-Lisans Programları Yatay Geçiş Yönergesi R3.pdf",
    "YÖ-0024 Öğrenci Toplulukları Kuruluş ve İşleyiş Yönergesi R4.pdf",
    "YÖ-0011 Uluslararası Öğrencilerin Lisans Programlarına Başvuru, Kabul ve Kayıt Yönergesi R7.pdf",
    "YN-0015 Yaz Öğretimi Yönetmeliği R1.pdf",
    "YO-0071_Temel_Bilimler_Fakultesi_Lisans_Eitimi_Staj_Yonergesi_R0.pdf",
    "YÖ-0035_Öğrenci_Konseyi_Yonergesi R1.pdf",
    "YÖ-0007 Lisans Muafiyet ve İntibak Yönergesi R2.pdf",
    "YÖ-0006 Mezuniyet Belgesi İle Diploma ve Diploma Defterinin Düzenlenmesinde Uyulacak Esaslara İlişkin Yönerge R2.pdf",
    "YÖ-0042 Erasmus Öğrenci ve Personel Değişim Yönergesi R3.pdf",
    "YO-0003_Dereceye_Giren_Lisans_Mezunlarn_Tespitine_likin_Yonerge_R1.pdf",
    "YO-0018_Mimarlk_Fakultesi_Lisans_Eitimi_Staj_Yonergesi_R2.pdf",
    "YÖ-0017 Mimari Tasarm VIII Dersi Uygulama Esasları R2.pdf",
    "YÖ-0016 Mimari Tasarım Dersleri Uygulama Esasları R2.pdf"
]

output_texts = []

for pdf_file in pdf_files:
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        is_first_page = True
        
        for page in pdf.pages:
            page_text = page.extract_text()
            lines = page_text.split('\n')
            filtered_lines = []
            
            if is_first_page:
                skip_until_content = False
                header_section = True
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('Doküman No'):
                        skip_until_content = True
                        continue
                    elif 'Sayfa' in line_stripped and any(c.isdigit() for c in line_stripped):
                        skip_until_content = False
                        header_section = False
                        continue
                    elif line_stripped.startswith('Form No:'):
                        continue
                    elif line_stripped.replace('.','').isdigit() and len(line_stripped) <= 4:
                        continue
                    
                    if header_section:
                        words = line_stripped.split()
                        upper_words = [word for word in words if word.isupper() and len(word) > 1]
                        if upper_words:
                            filtered_lines.append(' '.join(upper_words))
                    elif not skip_until_content:
                        filtered_lines.append(line)
                is_first_page = False
            else:
                skip_until_content = False
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    if line_stripped.startswith('Form No:'):
                        continue
                    elif line_stripped.startswith('Doküman No'):
                        skip_until_content = True
                        continue
                    elif 'Sayfa' in line_stripped and any(c.isdigit() for c in line_stripped):
                        skip_until_content = False
                        continue
                    elif line_stripped.replace('.','').isdigit() and len(line_stripped) <= 4:
                        continue
                    
                    if not skip_until_content:
                        filtered_lines.append(line)
            
            text += '\n'.join(filtered_lines) + '\n'
        output_texts.append(text)

with open("merged_text.txt", "w", encoding="utf-8") as f:
    for text in output_texts:
        f.write(text + "\n")
