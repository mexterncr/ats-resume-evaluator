from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

app = Flask(__name__)

TECH_SKILLS_DB = {
    "python", "java", "c++", "c", "javascript", "typescript", "html", "css", "sql", 
    "mysql", "postgresql", "mongodb", "react", "angular", "vue", "django", "flask", 
    "fastapi", "spring boot", "node.js", "express", "aws", "docker", "kubernetes", 
    "git", "github", "linux", "rest api", "machine learning", "deep learning", "nlp",
    "scikit-learn", "tensorflow", "pytorch", "pandas", "numpy", "data structures"
}

SECTIONS_TO_CHECK = {
    "Contact Info": [r"email", r"phone", r"linkedin", r"github"],
    "Education": [r"education", r"academic", r"bachelor", r"degree", r"university"],
    "Technical Skills": [r"skills", r"technical skills", r"technologies"],
    "Projects": [r"projects", r"academic projects", r"portfolio"],
    "Experience": [r"experience", r"employment", r"internship"],
    "Certifications": [r"certifications", r"courses", r"achievements"]
}

SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_file_content(file):
    filename = file.filename.lower()
    extracted_text = ""
    try:
        if filename.endswith('.pdf'):
            reader = PdfReader(file)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    extracted_text += txt + " "
        elif filename.endswith(('.docx', '.doc')):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                if para.text:
                    extracted_text += para.text + " "
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text:
                            extracted_text += cell.text + " "
        elif filename.endswith('.txt'):
            extracted_text = file.read().decode('utf-8', errors='ignore')
    except Exception:
        return ""
    return extracted_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    job_desc = request.form.get('job_desc', '').strip()
    resume_files = request.files.getlist('resume_file')
    is_demo = request.form.get('is_demo') == 'true'

    if not job_desc:
        return jsonify({'error': 'Please provide a target Job Description to continue.'}), 400

    clean_jd = clean_text(job_desc)
    resumes_data = []

    if is_demo:
        demo_text = """
        Mohammed CS Student. Email: student@email.com. Phone: +91 9876543210.
        Education: Bachelor of Science in Computer Science, 2026.
        Technical Skills: Python, SQL, C++, HTML, CSS, JavaScript, Flask, Git, GitHub, Machine Learning, Data Structures.
        Projects: AI Resume ATS Evaluator using Python NLP, Web Inventory System using Node.js and SQL.
        Certifications: Python Data Science.
        """
        resumes_data.append(("Sample_CS_Resume.pdf", demo_text))
    else:
        valid_files = [f for f in resume_files if f and f.filename.lower().endswith(SUPPORTED_EXTENSIONS)]
        if not valid_files:
            return jsonify({'error': 'Please upload a valid resume file (.PDF, .DOCX, or .TXT).'}), 400

        for file in valid_files:
            text = extract_file_content(file)
            if text.strip():
                resumes_data.append((file.filename, text))

        if not resumes_data:
            return jsonify({'error': 'Unable to parse text from the uploaded document(s). Please verify file permissions.'}), 400

    results = []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    for fname, r_text in resumes_data:
        clean_res = clean_text(r_text)

        # 1. TF-IDF & Cosine Similarity
        tfidf_matrix = vectorizer.fit_transform([clean_jd, clean_res])
        similarity = round(float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]) * 100, 1)

        # 2. Sections
        sections = {}
        for name, kws in SECTIONS_TO_CHECK.items():
            sections[name] = any(re.search(rf"\b{kw}\b", clean_res) for kw in kws)
        sec_score = round((sum(sections.values()) / len(sections)) * 100, 1)

        # 3. Skills
        jd_skills = {s for s in TECH_SKILLS_DB if re.search(rf"\b{re.escape(s)}\b", clean_jd)}
        res_skills = {s for s in TECH_SKILLS_DB if re.search(rf"\b{re.escape(s)}\b", clean_res)}
        
        matched = sorted(list(jd_skills.intersection(res_skills)))
        missing = sorted(list(jd_skills.difference(res_skills)))
        skill_score = round((len(matched) / len(jd_skills) * 100), 1) if jd_skills else 100.0

        ats_score = round((similarity * 0.40) + (skill_score * 0.40) + (sec_score * 0.20), 1)

        # Real-World HR Cutoffs
        if ats_score >= 60.0:
            status = "Shortlisted"
        elif ats_score >= 45.0:
            status = "Potential Match"
        else:
            status = "Rejected"

        results.append({
            'filename': fname,
            'timestamp': datetime.now().strftime('%d %b %Y, %I:%M %p'),
            'ats_score': ats_score,
            'status': status,
            'similarity': similarity,
            'skill_score': skill_score,
            'sec_score': sec_score,
            'matched_skills': matched,
            'missing_skills': missing,
            'sections': sections
        })

    results = sorted(results, key=lambda x: x['ats_score'], reverse=True)

    if len(results) == 1:
        return jsonify(results[0])
    return jsonify({'multiple': True, 'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
