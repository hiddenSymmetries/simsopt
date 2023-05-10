def importCurves(file):
    '''
    Reads a coils file and returns an array of shape (Ncoils,Npoints,3) in cartesian coordinates

    args:
        file: file to read.
    '''
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 

    coilN = 0
    coilPos = [[]]
    for nVals in range(len(allCoilsValues)):
        vals = allCoilsValues[nVals].split()
        try:
            floatVals = [float(nVals) for nVals in vals][0:3]
            coilPos[coilN].append(floatVals)
        except:
            try:
                floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
                coilPos[coilN].append(floatVals)
                coilN = coilN+1
                coilPos.append([])
            except:
                break

    return coilPos[:-1]


def importCoils_and_current(file):
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 

    coilN = 0
    coilPos = [[]]
    current = []
    flag = True
    for nVals in range(len(allCoilsValues)):
        vals = allCoilsValues[nVals].split()
        try:
            floatVals = [float(nVals) for nVals in vals][0:3]
            coilPos[coilN].append(floatVals)
            if flag:
                current.append(float(vals[3]))
                flag = False
        except:
            try:
                floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
                coilPos[coilN].append(floatVals)
                coilN = coilN+1
                coilPos.append([])
                flag = True
            except:
                break
    return coilPos[:-1], current


def import_current(file):
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 

    current = []
    flag = True
    for nVals in range(len(allCoilsValues)-1):
        vals = allCoilsValues[nVals].split()
        if flag:
            current.append(float(vals[3]))
            flag = False
        if len(vals) > 4:
            flag = True

    return current
