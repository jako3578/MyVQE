from MyVQE_V2 import*

HH = MyVQE(["H", "H"], ["0 0 0", "1.4 0 0"], basis="STO-3G")
HH.run(0.01, 5)
HH.currentEnergy

outfile = open("Out.txt", "a")

outfile.write(f"{HH.currentEnergy}")







