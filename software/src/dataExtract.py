import torchaudio

if __name__=="__main__":
    print("Downloading Speech Commands dataset...")
    torchaudio.datasets.SPEECHCOMMANDS(root="./data",download=True,subset="training")
    torchaudio.datasets.SPEECHCOMMANDS(root="./data",download=True,subset="validation")
    torchaudio.datasets.SPEECHCOMMANDS(root="./data",download=True,subset="testing")
    print("Done")
