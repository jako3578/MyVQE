from MyVQE_V2 import*
distances = np.linspace(1.18, 6.6, 12)
file = open("LiH.txt", "a")
file.write("# Calculations of ground state energy of LiH, depending on interatomic distance")
file.write("# distance | energy")
file.close()
for distance in distances:
    file = open("LiH.txt", "a")
    LiH = MyVQE(["Li", "H"], ["0 0 0", f"{distance} 0 0"], basis="STO-3G")
    LiH.run(0.01, 5)
    file.write(f"{distance} {LiH.currentEnergy}")
    file.close()
file.close()









