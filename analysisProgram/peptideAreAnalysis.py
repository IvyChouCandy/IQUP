import pandas as pd

file = "NCI"
path = f"D:/IQUPexperiment/PipetideAnalysis/{file}/"
filename = "preQRPARE.xlsx"
predictFilename = "proteinpeptide.csv"

proteinAreDf = pd.read_excel(f"{path}{filename}")
pictureAreDf = pd.read_csv(f"{path}{predictFilename}")

filter = (proteinAreDf["change"] == 0)
areZeroDf = proteinAreDf[filter]
areZeroProteinpeptide = set(areZeroDf["Protein_Peptide"].values.tolist())
areZeroPictureDf = pictureAreDf[[True if i in areZeroProteinpeptide else False for i in pictureAreDf["Protein_Peptide"]]]
areZeroPictureDf.to_csv(f"{path}PeptideAreZeroPicture.csv")
