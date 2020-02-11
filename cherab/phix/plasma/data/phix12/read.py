#create = open("~/Documents/coil.dat","w")
with open("new/coil_new.dat","w") as file: pass

keywords='kappa'
with open("coil.dat", "r") as ins:

    #for line in ins:
        #if keywords in line:
            #kappa = float(line.split()[3])
            #print(kappa*3)

    for line in ins:
        #with open("/c/Users/'WANG JINGTING'/Documents/coil.dat","a") as mon:
        #with open("~/Documents/coil.dat","a") as mon:
        with open("new/coil_new.dat","a") as mon:
            mon.write(line)
