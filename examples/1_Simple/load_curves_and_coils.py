from simsopt.geo import CurveXYZFourier
from simsopt.field import  Coil, Current
from simsopt.field.coil import coils_via_file , coils_to_makegrid
from simsopt.util.coil_util import *

file = "inputs/coils.tester"
#file = "inputs/coils.w7x_std_mc"
#file = "inputs/coils.bigtest"
#file = "inputs/tester.dat"
#------------------------------------------------------------------------------
#curves = CurveXYZFourier.load_curves_from_coils_file(file, 19,ppp=5)
#coils  = coils_via_file(file, 19,5)
#
#currents = [i.current.get_value() for i in coils]
#print(len(curves))
#print(currents)
#curves_from_coil  = [i.curve for i in coils]
#print(len(curves_from_coil))
##Coil.export_coils_in_cartesian("output_with_coil", coils, NFP=1)
#Coil.export_coils_in_cartesian("output_with_curves_from_coil&current", curves_from_coil,currents, 1)
#Coil.export_coils("output_with_curves_from_coil&current2", coils, 1)
#Coil.export_coils_in_cartesian("output_with_curve&current", curves,currents, 1)
#print(import_current(file))
##print(importCoils_and_current(file))

#---------------------------------------------------------------------------------------
#curves_old = CurveXYZFourier.load_curves_from_coils_file(file, 19,ppp=5)
#curves_new = CurveXYZFourier.load_curves_from_file_new(file, 19,ppp=5,Cartesian=True)
#currents = import_current(file)
#
#Coil.export_coils_in_cartesian("output_new", curves_new,currents, 1)
#Coil.export_coils_in_cartesian("output_old", curves_old,currents, 1)
#print(importCurves("output_new") == importCurves("output_old"))
#---------------------------------------------------------------------------------------
#curves_old = CurveXYZFourier.load_curves_from_file(file, 19,ppp=5)
#curves_new = CurveXYZFourier.load_curves_from_file_new(file, 19,ppp=5)
#print(len(curves_old))
#currents = [1]*len(curves_old)
#print(currents)
#Coil.export_coils_in_cartesian("output_new", curves_new,currents, 1)
#Coil.export_coils_in_cartesian("output_old", curves_old,currents, 1)
#print(importCurves("output_new") == importCurves("output_old"))
#---------------------------------------------------------------------------------------
#coils_old = coils_via_file(file,19,ppp=5)
#coils_new = coils_via_file_new(file, 19,ppp=5)
#
#Coil.export_coils("output_new", coils_new, 1)
#Coil.export_coils("output_old", coils_old, 1)
#print(importCurves("output_new") == importCurves("output_old"))
#--------------------------------------------------------------------------------------
curves_old = CurveXYZFourier.load_curves_from_file(file, 19,ppp=5,Cartesian=True)
#curves_new = CurveXYZFourier.load_curves_from_file_new(file, 19,ppp=5,Cartesian=True)

currents = [Current(1)]*len(curves_old)

coils_to_makegrid("output_old", curves_old,currents)
#coils_to_makegrid("output_new", curves_new,currents)

print(importCurves("output_new") == importCurves("output_old"))