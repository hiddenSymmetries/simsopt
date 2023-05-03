from simsopt.geo import CurveXYZFourier
from simsopt.field import coils_via_file, Coil
from simsopt.util.coil_util import *

file = "inputs/coils.tester"
#file = "inputs/coils.w7x_std_mc"
#file = "inputs/coils.bigtest"

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

#-------------------------------------------------------------------------------
