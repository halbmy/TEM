# %%
import utm
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stem import readSettings, sTEMRhoModelling
# from stem import sTEMBlockModelling
import pygimli as pg

def rhoa(t, dbzdt, m=1.0):
    mu0 = 4e-7*np.pi
    return mu0 / np.pi * (mu0 * m / 20.0)**(2/3) * np.abs(dbzdt)**(-2/3) * t**(-5/3)

# dr = dr/du * du = -2/3 r / u *du => |dr/r| = 2/3 |dr/r|

# %%
class TEM:
    """Class for processing TEM data."""
    def __init__(self, filename:str=None, cfg=None, **kwargs):
        self.thk = kwargs.pop("thk", np.arange(2, 28, 2))
        if cfg is None and filename.endswith(".xyz"):
            if Path(filename[:-4]+".gex").exists():
                cfg = filename[:-4]+".gex"
        if cfg is not None:
            if isinstance(cfg, str):
                self.readCFG(cfg)
            else:
                self.cfg = cfg
        if filename is not None:
            self.readData(filename)

    def __str__(self):
        """String representation."""
        return f"TEM profile with {len(self.DATA):d} soundings\n" + \
            f"{len(self.t):d} times ({min(self.t)}-{max(self.t)})"

    def readCFG(self, cfg):
        """Read data file (xyz)."""
        self.cfg = readSettings(cfg)
        self.f = sTEMRhoModelling(thk=self.thk, cfg=self.cfg)
        self.t = self.f.t

    def showWaveform(self):
        """Show current waveform."""
        fig, ax = plt.subplots()
        ax.plot(self.cfg["tL"], self.cfg["vL"], label="LM")
        ax.plot(self.cfg["tH"], self.cfg["vH"], label="HM")
        ax.set_xlabel("t [s]")
        ax.legend()
        ax.grid()
        return ax

    def readData(self, filename):
        """Read data file (xyz)."""
        self.data = pd.read_csv(filename, skiprows=2, sep=r"\s+")
        self.extractData()

    def extractData(self):
        """Extract data from dataframe into arrays."""
        nL = len(self.cfg["timeL"])
        CH1 = np.column_stack([self.data[f"dbdt{i:03d}_Ch1"] for i in range(1, nL+1)])
        SD1 = np.column_stack([self.data[f"stdF{i:03d}_Ch1"] for i in range(1, nL+1)])
        nH = len(self.cfg["timeH"])
        CH2 = np.column_stack([self.data[f"dbdt{i:03d}_Ch2"] for i in range(1, nH+1)])
        SD2 = np.column_stack([self.data[f"stdF{i:03d}_Ch2"] for i in range(1, nH+1)])
        self.DATA = np.hstack([CH1, CH2])
        self.SD = np.hstack([SD1, SD2])
        self.calcRhoa()

    def calcRhoa(self, rmin=1, rmax=10000):
        """Compute apparent resistivity."""
        self.RHOA = np.zeros_like(self.DATA)
        for n, data in enumerate(self.DATA):
            self.RHOA[n] = rhoa(self.t, data)

        self.RHOA[self.RHOA < rmin] = np.nan
        self.RHOA[self.RHOA > rmax] = np.nan

    def showRhoa(self, rmin=10, rmax=500, **kwargs):
        """Show apparent resistivity."""
        kwargs.setdefault("cmap", "Spectral_r")
        plt.imshow(np.log10(self.RHOA.T),
                   vmin=np.log10(rmin), vmax=np.log10(rmax),
                   **kwargs)

    def showSounding(self, n=0, rhoa=False, ax=None, **kwargs):
        """Show single sounding."""
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("ls", ":")
        if ax is None:
            fig, ax = plt.subplots()

        data = self.DATA[n]
        err = self.SD[n]*np.abs(data)
        if rhoa:
            data = self.RHOA[n]
            err = 2/3 * self.SD[n]*np.abs(data)

        n0 = 0
        alln = np.hstack([np.nonzero(np.diff(self.t)<0)[0]+1, len(self.t)])
        kwargs.setdefault("color", None)
        for nn in alln:
            er = ax.errorbar(self.t[n0:nn], data[n0:nn], yerr=err[n0:nn], **kwargs)
            if kwargs["color"] is None:
                kwargs["color"] = er.lines[0].get_color()
            n0 = nn

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        return ax

    def showSoundings(self, nn=None, **kwargs):
        """Show several soundings in one figure."""
        if nn is None:
            nn = range(self.DATA.shape[0])
        ax = None
        for i in nn:
            ax = self.showSounding(i, ax=ax, **kwargs)

        return ax

    def showPositions(self):
        """Show sounding positions."""
        fig, ax = plt.subplots()
        x, y, *_ = utm.from_latlon(self.data["Latitude"].to_numpy(),
                                   self.data["Longitude"].to_numpy())
        ax.plot(x, y, "x")
        for i in range(0, len(x), 10):
            ax.text(x[i], y[i], str(i), va="center", ha="center")

        ax.set_aspect(1.0)
        ax.grid(True)
        return ax

    def filter(self, tmin=0, tmax=1e9, nmin=0, nmax=9999, n=None):
        """Filter data."""
        doex = False
        if nmin > 0:
            self.data.drop(np.nonzero(self.data.index < nmin)[0], inplace=True)
            doex = True
        if nmax < self.DATA.shape[0]:
            self.data.drop(np.nonzero(self.data.index > nmax)[0], inplace=True)
            doex = True
        if n is not None:
            self.data.drop(n, inplace=True)
            doex = True
        if doex:
            self.extractData()

    def invertSounding(self, n=0, show=True):
        """Invert data."""
        inv = pg.Inversion(fop=self.f)
        # inv.dataTrans = "log" # not really necessary
        data = np.abs(self.DATA[n])
        err = np.maximum(self.SD[n], 0.015)
        err[self.DATA[n] <1e-15] = 1e8
        err[err > 1] = 1e8
        self.res1d = inv.run(data, err, verbose=1, startModel=self.f.createStartVector(data))
        self.inv1d = inv
        if show:
            self.show1dResult()

    def show1dResult(self):
        """Show 1D model and data fit."""
        inv = self.inv1d
        data = inv.dataVals
        fig, ax = plt.subplots(ncols=2)
        pg.viewer.mpl.drawModel1D(ax[0], self.thk, self.res1d, plot="semilogx")
        ax[0].invert_yaxis()
        ax[1].errorbar(self.t, data, yerr=inv.errorVals*data, marker="+", ls="None", label="data")
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        n0 = 0
        alln = np.hstack([np.nonzero(np.diff(self.t)<0)[0]+1, len(self.t)])
        color = None
        for nn in alln:
            li = ax[1].plot(self.t[n0:nn], inv.response[n0:nn], color=color)[0]
            color = li.get_color() if color is None else None
            n0 = nn

        ax[1].grid()

    def invertAll(self, show=True):
        """Invert all data individually."""
        self.invertSounding(0, show=False)
        res = self.res1d
        self.MODELS = [res]
        self.CHI2 = [self.inv1d.chi2()]
        for n in range(1, len(self.DATA)):
            data = np.abs(self.DATA[n])
            err = np.maximum(self.SD[n], 0.01)
            err[self.DATA[n] <1e-15] = 1e8
            err[err > 1] = 1e8
            res = self.inv1d.run(data, err, verbose=0, startModel=res)
            self.MODELS.append(res)
            self.CHI2.append(self.inv1d.chi2())
            pg.info(f"{n}: chi2={self.CHI2[-1]:.2f} ({self.inv1d.iter:d} iter)")

        if show:
            self.showResults()

    def invertLCI(self, **kwargs):
        """Invert all data with lateral constraints."""
        pass

    def invertSCI(self, **kwargs):
        """Invert all data with lateral constraints."""
        pass

    def showResults(self, usepos=True, **kwargs):
        kwargs.setdefault("cMin", 10)
        kwargs.setdefault("cMin", 500)
        kwargs.setdefault("cMap", "Spectral_r")
        if usepos is not None:
            x, y, *_ = utm.from_latlon(self.data["Latitude"].to_numpy(),
                                    self.data["Longitude"].to_numpy())
            if usepos == "x": # Easting
                kwargs["x"] = x
            elif usepos == "y": # Northing
                kwargs["x"] = y
            else: # tape measure
                dx = np.sqrt(np.diff(x)**2 +np.diff(y)**2)
                kwargs["x"] = np.hstack([0, np.cumsum(dx)])
        if "Elevation" in self.data:
            kwargs.setdefault("topo", self.data["Elevation"].to_numpy())
        return pg.viewer.mpl.showStitchedModels(
            self.MODELS, thk=self.thk, **kwargs)

# %%
if __name__ == "__main__":
    self = TEM("data/Madsen2026/sTEM2.xyz",
               cfg="data/Madsen2026/sTEM.gex")
    print(self)
    # %%
    ax = self.showWaveform()
    ax.set_xlim(-1e-7, 5e-6)
    ax.set_ylim(-0.001, 1.001)
    # %%
    self.filter(nmax=20)
    self.filter(n=[9, 10])
    # self.showPositions()
    # %%
    # self.showRhoa(rmax=100)
    # self.invertSounding()
    # self.invertAll()
    # %%
    # f = pg.frameworks.MultiFrameModelling(self.f)