#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

matplotlib.use("QtAgg")


def main(file, printinfo, pltdata, pltconts, plt3d, s_plot_ignore=0.3):
    print("usage: vmecPlot <woutXXX.nc>")

    import math
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    ################################################################################
    ################################ READ VMEC DATA ################################
    ################################################################################
    if 1 == 1:
        fname = file
        f = netcdf_file(fname, "r", mmap=False)
        print("-------------------")
        print(list(f.variables.keys()))
        print("-------------------")

        phi = f.variables["phi"][()]
        iotaf = f.variables["iotaf"][()]
        presf = f.variables["presf"][()]
        iotas = f.variables["iotas"][()]
        pres = f.variables["pres"][()]
        ns = f.variables["ns"][()]
        nfp = f.variables["nfp"][()]
        xn = f.variables["xn"][()]
        xm = f.variables["xm"][()]
        xn_nyq = f.variables["xn_nyq"][()]
        xm_nyq = f.variables["xm_nyq"][()]
        rmnc = f.variables["rmnc"][()]
        zmns = f.variables["zmns"][()]
        bmnc = f.variables["bmnc"][()]
        raxis_cc = f.variables["raxis_cc"][()]
        zaxis_cs = f.variables["zaxis_cs"][()]
        buco = f.variables["buco"][()]
        bvco = f.variables["bvco"][()]
        jcuru = f.variables["jcuru"][()]
        jcurv = f.variables["jcurv"][()]
        lasym = f.variables["lasym__logical__"][()]
        if lasym == 1:
            rmns = f.variables["rmns"][()]
            zmnc = f.variables["zmnc"][()]
            bmns = f.variables["bmns"][()]
            raxis_cs = f.variables["raxis_cs"][()]
            zaxis_cc = f.variables["zaxis_cc"][()]
        else:
            rmns = 0 * rmnc
            zmnc = 0 * rmnc
            bmns = 0 * bmnc
            raxis_cs = 0 * raxis_cc
            zaxis_cc = 0 * raxis_cc

        try:
            ac = f.variables["ac"][()]
        except:
            ac = []

        try:
            pcurr_type = f.variables["pcurr_type"][()]
        except:
            pcurr_type = ""

        mpol = f.variables["mpol"][()]
        ntor = f.variables["ntor"][()]
        Aminor_p = f.variables["Aminor_p"][()]
        Rmajor_p = f.variables["Rmajor_p"][()]
        aspect = f.variables["aspect"][()]
        betatotal = f.variables["betatotal"][()]
        betapol = f.variables["betapol"][()]
        betator = f.variables["betator"][()]
        betaxis = f.variables["betaxis"][()]
        ctor = f.variables["ctor"][()]
        DMerc = f.variables["DMerc"][()]
        gmnc = f.variables["gmnc"][()]

        dVds = 4 * np.pi * np.pi * np.abs(gmnc[1:, 0])
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
        av_well = (dVds_s0 - dVds_s1) / dVds_s0

        wells = []
        for i in range(ns - 3):
            dVds_s0 = 1.5 * dVds[i] - 0.5 * dVds[i + 1]
            dVds_s1 = 1.5 * dVds[i + 1] - 0.5 * dVds[i + 2]
            well = (dVds_s0 - dVds_s1) / dVds_s0
            wells = np.append(wells, well)

    ################################################################################
    ################################ PRINT VMEC DATA ################################
    ################################################################################
    if printinfo == True:
        print("nfp: ", nfp)
        print("ns: ", ns)
        print("mpol: ", mpol)
        print("ntor: ", ntor)
        print("Aminor_p: ", Aminor_p)
        print("Rmajor_p: ", Rmajor_p)
        # print("aspect:            ",data)
        print("Rmajor_p/Aminor_p: ", Rmajor_p / Aminor_p)
        print("betatotal: ", betatotal)
        print("betapol:   ", betapol)
        print("betator:   ", betator)
        print("betaxis:   ", betaxis)
        print("ctor:   ", ctor)
        print("Avg Well =", av_well)
        print("wells =", wells)
    f.close()

    printinput = False
    if printinput == True:
        print("PHIEDGE =", phi[-1])
        print("PRES_F =", presf)
        # print("rmnc=",rmnc[-1,:])
        print("raxis_cc=", raxis_cc)
        # print("zmns=",zmns[-1,:])
        print("zaxis_cs=", zaxis_cs)
        rbc = rmnc[-1, :]
        zbs = zmns[-1, :]
        mpolarr = np.linspace(-mpol + 1, mpol - 1, 2 * (mpol - 1) + 1)
        ntorarr = np.linspace(0, ntor - 1, ntor)

        for i in range(len(rbc)):
            if rbc[i] != 0 or zbs[i] != 0:
                print(
                    f"RBC({int(xn[i] / nfp):2d},{int(xm[i]):2d})={rbc[i]:24.15e}, ZBS({int(xn[i] / nfp):2d},{int(xm[i]):2d})={zbs[i]:24.15e}"
                )

    ################################################################################
    ################################# CALCULATIONS #################################
    ################################################################################
    if 1 == 1:
        nmodes = len(xn)
        s = np.linspace(0, 1, ns)
        s_half = [(i - 0.5) / (ns - 1) for i in range(1, ns)]
        phiedge = phi[-1]
        phi_half = [(i - 0.5) * phiedge / (ns - 1) for i in range(1, ns)]
        ntheta = 200
        nzeta = 8
        theta = np.linspace(0, 2 * np.pi, num=ntheta)
        zeta = np.linspace(0, 2 * np.pi / nfp, num=nzeta, endpoint=False)
        iradius = ns - 1
        R = np.zeros((ntheta, nzeta))
        Z = np.zeros((ntheta, nzeta))
        for itheta in range(ntheta):
            for izeta in range(nzeta):
                for imode in range(nmodes):
                    angle = xm[imode] * theta[itheta] - xn[imode] * zeta[izeta]
                    R[itheta, izeta] = (
                        R[itheta, izeta]
                        + rmnc[iradius, imode] * math.cos(angle)
                        + rmns[iradius, imode] * math.sin(angle)
                    )
                    Z[itheta, izeta] = (
                        Z[itheta, izeta]
                        + zmns[iradius, imode] * math.sin(angle)
                        + zmnc[iradius, imode] * math.cos(angle)
                    )

        Raxis = np.zeros(nzeta)
        Zaxis = np.zeros(nzeta)
        for izeta in range(nzeta):
            for n in range(ntor + 1):
                angle = -n * nfp * zeta[izeta]
                Raxis[izeta] += raxis_cc[n] * math.cos(angle) + raxis_cs[
                    n
                ] * math.sin(angle)
                Zaxis[izeta] += zaxis_cs[n] * math.sin(angle) + zaxis_cc[
                    n
                ] * math.cos(angle)

    ################################################################################
    ################################ PLOT VMEC DATA ################################
    ################################################################################
    if pltdata == True:
        xLabel = r"$s = \psi_N$"

        fig = plt.figure("VMEC Data", figsize=(14, 7))
        fig.patch.set_facecolor("white")

        numCols = 3
        numRows = 3
        plotNum = 1

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi, iotaf, '.-',label='iotaf')
        # plt.plot(phi_half, iotas[1:],'.-',label='iotas')
        plt.plot(s, iotaf, ".-", label="iotaf")
        plt.plot(s_half, iotas[1:], ".-", label="iotas")
        plt.legend(fontsize="x-small")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi, presf, '.-',label='presf')
        # plt.plot(phi_half, pres[1:], '.-',label='pres')
        plt.plot(s, presf, ".-", label="presf")
        plt.plot(s_half, pres[1:], ".-", label="pres")
        plt.legend(fontsize="x-small")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi_half, buco[1:], '.-',label='buco')
        plt.plot(s_half, buco[1:], ".-", label="buco")
        plt.title("buco")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi_half, bvco[1:], '.-',label='bvco')
        plt.plot(s_half, bvco[1:], ".-", label="bvco")
        plt.title("bvco")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi, jcuru, '.-',label='jcuru')
        plt.plot(s, jcuru, ".-", label="jcuru")
        plt.title("jcuru")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # plt.plot(phi, jcurv, '.-',label='jcurv')
        plt.plot(s, jcurv, ".-", label="jcurv")
        plt.title("jcurv")
        plt.xlabel(xLabel)

        plt.subplot(numRows, numCols, plotNum)
        plotNum += 1
        # if 'power_series' in pcurr_type:
        # 	ac_profile = phi*0.0
        # 	for i in range(len(ac)):
        # 		ac_profile += ac[i]*(s**i)
        # 	plt.plot(s, ac_profile, '.-')
        # else:
        # 	mask = (ac_aux_s >= 0)
        # 	plt.plot(ac_aux_s[mask], ac_aux_f[mask],'.-')
        # plt.title('ac profile')
        # plt.xlabel(xLabel)
        plt.plot(
            s[int(s_plot_ignore * len(s)) : -2],
            DMerc[int(s_plot_ignore * len(s)) : -2],
            ".-",
        )
        plt.title("DMerc")
        plt.xlabel(xLabel)
        # if save == True:
        # plt.savefig(file[:-3]+'_VMECparams.pdf', bbox_inches = 'tight', pad_inches = 0)

    ################################################################################
    ################################ PLOT CONTOURS #################################
    ################################################################################
    if pltconts == True:
        Ntheta = 351
        Nphi = 351
        thetas = np.linspace(0, 2 * np.pi, Ntheta)
        phis = np.linspace(0, 2 * np.pi / nfp, Nphi)
        phis2D, thetas2D = np.meshgrid(phis, thetas)

        sarr = [1, ns / 4, ns / 2, ns - 2]
        bbooz(fname, sarr, ns)
        # if save == True:
        #    plt.savefig(file[:-3]+'_boozcontours'.pdf', bbox_inches = 'tight', pad_inches = 0)

    ################################################################################
    ################################ POINCARE PLOTS ################################
    ################################################################################
    if pltxsec == True:
        fig = plt.figure("Poincare Plots", figsize=(14, 7))
        fig.patch.set_facecolor("white")

        R = np.zeros((ntheta, nzeta))
        Z = np.zeros((ntheta, nzeta))
        for itheta in range(ntheta):
            for izeta in range(nzeta):
                for imode in range(nmodes):
                    angle = xm[imode] * theta[itheta] - xn[imode] * zeta[izeta]
                    R[itheta, izeta] = (
                        R[itheta, izeta]
                        + rmnc[iradius, imode] * math.cos(angle)
                        + rmns[iradius, imode] * math.sin(angle)
                    )
                    Z[itheta, izeta] = (
                        Z[itheta, izeta]
                        + zmns[iradius, imode] * math.sin(angle)
                        + zmnc[iradius, imode] * math.cos(angle)
                    )

        numCols = 3
        numRows = 2
        plotNum = 1

        plt.subplot(numRows, numCols, plotNum)
        # plt.subplot(1,1,1)
        plotNum += 1
        for ind in range(nzeta):
            plt.plot(R[:, ind], Z[:, ind], "-")
            plt.plot(R[:, ind], Z[:, ind], "-")
            plt.plot(R[:, ind], Z[:, ind], "-")
            plt.plot(R[:, ind], Z[:, ind], "-")
        plt.gca().set_aspect("equal", adjustable="box")
        # plt.legend(fontsize='x-small')
        plt.xlabel("R")
        plt.ylabel("Z")

        ntheta = 500
        nzeta = 5
        nradius = 10
        radii = np.linspace(1, ns - 1, nradius)
        radii = np.floor(radii)
        theta = np.linspace(0, 2 * np.pi, num=ntheta)
        zeta = np.linspace(0, 2 * np.pi / nfp / 2, num=nzeta, endpoint=True)

        from fractions import Fraction

        denoms = np.linspace(0, 1, num=nzeta)
        titles = []
        for idenom in range(len(denoms)):
            denom = denoms[idenom]
            anglestr = (
                str(Fraction(denom).limit_denominator().numerator)
                + r"$\pi/$"
                + str(Fraction(denom).limit_denominator().denominator)
            )
            if anglestr[:7] == "1$\pi/$":
                anglestr = "$\pi/$" + anglestr[7:]
            titles.append(r"$\phi=$" + anglestr)
        titles[0] = r"$\phi=0$"
        titles[-1] = r"$\phi=\pi$"

        def FindBoundary(theta, phi, iradius):
            angle = xm * theta + xn * phi
            rb = np.sum(rmnc[iradius, :] * np.cos(angle))
            zb = np.sum(zmns[iradius, :] * np.sin(angle))
            if lasym == True:
                rb += np.sum(rmns[iradius, :] * np.sin(angle))
                zb += np.sum(zmnc[iradius, :] * np.cos(angle))
            return rb, zb

        iradii = np.linspace(0, ns - 1, num=nradius).round()
        iradii = [int(i) for i in iradii]
        R = np.zeros((ntheta, nzeta, nradius))
        Z = np.zeros((ntheta, nzeta, nradius))
        for itheta in range(ntheta):
            for izeta in range(nzeta):
                for iradius in range(len(radii)):
                    rad = int(radii[iradius])
                    R[itheta, izeta, iradius], Z[itheta, izeta, iradius] = (
                        FindBoundary(theta[itheta], zeta[izeta], rad)
                    )

        Raxis = np.zeros(nzeta)
        Zaxis = np.zeros(nzeta)
        for jn in range(len(raxis_cc)):
            n = jn * nfp
            sinangle = np.sin(n * zeta)
            cosangle = np.cos(n * zeta)

            Raxis += raxis_cc[jn] * cosangle
            Zaxis += zaxis_cs[jn] * sinangle

        for izeta in range(nzeta):
            plt.subplot(numRows, numCols, plotNum)
            plotNum += 1
            for iradius in range(nradius):
                plt.plot(R[:, izeta, iradius], Z[:, izeta, iradius], "-")
            plt.plot(Raxis[izeta], Zaxis[izeta], "xr")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("R")
            plt.ylabel("Z")
            plt.title((titles[izeta]))

        plt.subplots_adjust(wspace=0.39, hspace=0.444)
        plt.figtext(
            0.5, 0.99, os.path.abspath(fname), ha="center", va="top", fontsize=6
        )

        # maximizeWindow()

        # plt.tight_layout()
        plt.figtext(
            0.5, 0.99, os.path.abspath(fname), ha="center", va="top", fontsize=6
        )
        # if save == True:
        #    plt.savefig(file[:-3]+'_PoincarePlots.pdf', bbox_inches = 'tight', pad_inches = 0)

    ################################################################################
    ################################ 3D SURFACE PLOT################################
    ################################################################################
    if plt3d == True:
        fig = plt.figure("3D Surface Plot")

        ntheta = 50
        nzeta = int(100 * nfp)
        theta1D = np.linspace(0, 2 * np.pi, num=ntheta)
        zeta1D = np.linspace(0, 2 * np.pi, num=nzeta)
        zeta2D, theta2D = np.meshgrid(zeta1D, theta1D)
        iradius = ns - 1
        R = np.zeros((ntheta, nzeta))
        Z = np.zeros((ntheta, nzeta))
        B = np.zeros((ntheta, nzeta))
        for imode in range(nmodes):
            angle = xm[imode] * theta2D - xn[imode] * zeta2D
            R = (
                R
                + rmnc[iradius, imode] * np.cos(angle)
                + rmns[iradius, imode] * np.sin(angle)
            )
            Z = (
                Z
                + zmns[iradius, imode] * np.sin(angle)
                + zmnc[iradius, imode] * np.cos(angle)
            )

        for imode in range(len(xn_nyq)):
            angle = xm_nyq[imode] * theta2D - xn_nyq[imode] * zeta2D
            B = (
                B
                + bmnc[iradius, imode] * np.cos(angle)
                + bmns[iradius, imode] * np.sin(angle)
            )

        X = R * np.cos(zeta2D)
        Y = R * np.sin(zeta2D)
        # Rescale to lie in [0,1]:
        B_rescaled = (B - B.min()) / (B.max() - B.min())

        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(projection="3d")
        ax._axis3don = False
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=cm.jet(B_rescaled),
            rstride=1,
            cstride=1,
            antialiased=False,
        )
        ax.auto_scale_xyz(
            [X.min(), X.max()], [X.min(), X.max()], [X.min(), X.max()]
        )

        plt.figtext(
            0.5, 0.99, os.path.abspath(fname), ha="center", va="top", fontsize=6
        )
        # if save == True:
        #    plt.savefig(file[:-3]+'_VMEC3Dplot.pdf', bbox_inches = 'tight', pad_inches = 0)

    plt.show()
    plt.close()


def bbooz(fname, sarr, ns):
    fig = plt.figure(figsize=(5, 3.2))
    import booz_xform as bx

    b = bx.Booz_xform()
    b.read_wout(fname)

    mpol = 20
    ntor = 20

    b.mboz = mpol
    b.nboz = ntor
    b.compute_surfs = sarr
    b.run()

    sarr = (np.array(sarr) + 2) / ns

    nphi = 100
    ntheta = 100

    nconts = 18

    axs = []

    nfp = b.nfp
    lasym = b.asym

    if nfp == 1:
        phimax_lab = "2$\pi$"
    elif nfp % 2 == 1:
        phimax_lab = "2$\pi$/" + str(int(nfp))
    else:
        if nfp == 2:
            phimax_lab = "$\pi$"
        else:
            phimax_lab = "$\pi$/" + str(int(nfp / 2))

    B = np.zeros((4, ntheta, nphi))

    for js in range(len(sarr)):
        s = sarr[js]

        if b.bmnc[1, 1] < 0:
            phimin = np.pi / nfp
        else:
            phimin = 0
        phimax = phimin + 2 * np.pi / nfp

        phis = np.linspace(0, 2 * np.pi / nfp, nphi) + phimin
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        phis2D, thetas2D = np.meshgrid(phis, thetas)

        # Extract relevant quantities from Boozer object
        # Mode numbers
        xm_nyq = b.xm_b
        xn_nyq = b.xn_b

        # Fourier coefficients
        bmnc = np.array(b.bmnc_b).T
        bmns = np.array(b.bmns_b).T

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            cosangle = np.cos(angle)
            B[js, :, :] += bmnc[js, jmn] * cosangle[:, :]
            if lasym == True:
                sinangle = np.sin(angle)
                B[js, :, :] += bmns[jmn] * sinangle[:, :]

    from matplotlib import cm

    for js in range(len(sarr)):
        s = sarr[js]
        plt.subplot(2, 2, js + 1)

        cplot = plt.contour(
            phis2D - phimin,
            thetas2D,
            B[js, :, :],
            nconts,
            linewidths=1.0,
            cmap=cm.plasma,
        )

        plt.colorbar()
        axs.append(plt.gca())
        plt.gca().tick_params(direction="in", length=0)
        color = (0.9, 0.9, 0.9)
        plt.text(
            0.05,
            6.13,
            f"|B| @ s={s}",
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.1",
                ec=color,
                fc=color,
            ),
        )
        plt.yticks([0, 2 * np.pi], ["0", "$2\pi$"])
        plt.ylabel(r"$\theta$", labelpad=-12)
        plt.xticks([0, 2 * np.pi / nfp], ["0", phimax_lab])
        plt.xlabel(r"$\varphi$", labelpad=-7)

    plt.subplots_adjust(
        left=0.054,
        bottom=0.07,
        right=0.94,
        top=0.982,
        wspace=0.39,
        hspace=0.244,
    )


def boozmnFor3D(fname, s):
    f = netcdf_file(fname, "r", mmap=False)
    s = f.variables["ns"][()] - 2

    import booz_xform as bx

    b1 = bx.Booz_xform()
    b1.read_wout(fname)
    b1.compute_surfs = [s]
    b1.mboz = 40
    b1.nboz = 40
    b1.run()
    b1.write_boozmn("boozmn_" + fname[:-3] + ".nc")


# Prints the VMEC data
printinfo = True
# Plots the VMEC data
pltdata = True
# Plots the |B| contours on four flux-surfaces
pltconts = True
# Plots the plasma cross-sections
pltxsec = True
# Plots the plasma boundary + |B| field
plt3d = True
# Saves the images as PDFs
save = False

file = "/Users/issraali/codes/simsopt/examples/2_Intermediate/wout_default_000_000830.nc"
#'/home/IPP-HGW/aliiss/projects/git/gloloc/plotting/wout_default_000_000000.nc'
#'/home/IPP-HGW/aliiss/projects/git/global-optimisation/scripts/wout_quasr_000_000000.nc'
#'/home/IPP-HGW/aliiss/projects/git/global-optimisation/wout_nfp4_QH_warm_start_000_000000.nc'


main(file, printinfo, pltdata, pltconts, plt3d, save)
