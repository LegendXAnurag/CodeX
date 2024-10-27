import numpy
def Dataset(actualMass,numPoints):
    #seed to function is used so that everytime the same datasets are returned
    numpy.random.seed(0)
    #adding errors in the force
    noiseLevel=0.2
    #generating uniformly distributed accn values
    acceleration=numpy.linspace(1,20,numPoints)
    #generating corresponding force with erros
    force = actualMass * acceleration + numpy.random.normal(0,noiseLevel,numPoints)
    return (acceleration,force)