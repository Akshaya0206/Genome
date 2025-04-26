# Gene Interaction Strength Prediction using SVM

This project predicts the interaction strength between genes — classified as **Strong**, **Moderate**, or **Weak** — using a **Support Vector Machine (SVM) classifier**.

##  Overview
- Built an SVM-based classification model on genomic data.
- Achieved a high accuracy of **99.24%**.
- Input features included support columns and p-value columns.
- Output classes: Strong interaction, Moderate interaction, Weak interaction.

## Technologies Used
- Python
- scikit-learn (SVM, model training and evaluation)
- Pandas (Data preprocessing)
- NumPy
- Matplotlib & Seaborn (optional for visualization)

##  How to Run
1. Clone this repository:
   git clone https://github.com/your-username/Genome.git
2. Navigate into the project directory:
   cd Genome
3. Install the required Python libraries:
   pip install pandas scikit-learn numpy
4. Run the training script:
   python train_model.py
5. Run the web application:
   python app.py
