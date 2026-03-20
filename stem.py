import numpy as np
import matplotlib.pyplot as plt
import empymod
import pygimli as pg

def readGEXFile(fname="sTEM.gex"):
    """Read GEX file (a rather general function)."""
    with open(fname) as fid:
        lines = fid.readlines()

    out = {}
    putin = out
    for line in lines:
        line = line.replace("\n", "").replace("\r", "")
        if len(line) > 2 and line[0] == "[" and line[-1] == "]":
            sect = line[1:-1]
            if sect != "General":
                putin = out[sect] = {}

        if "=" in line:
            nam, val = line.split("=")
            try:
                putin[nam] = np.fromstring(val, dtype=float, sep=" ")
            except ValueError:
                putin[nam] = val

    return out

def collectNumData(dic, name, start=1, stop=100, num=0):
    """Collect data from dictionary stored in numerically named keys."""
    if num == 0:
        for num in range(1, 4):
            if f"{name}{start:0{num}d}" in dic:
                break
    i = start
    col = []
    for i in range(start, stop):
        key = f"{name}{i:0{num}d}"
        if key in dic:
            col.append(dic[key])
        else:
            break

    return np.array(col)


def readSettings(filename="sTEM.gex"):
    """Read settings."""
    out = readGEXFile(filename)
    cfg = {}
    cfg["rxpos"] = out["RxCoilPosition1"]
    cfg["txpos"] = out["TxCoilPosition1"]
    cfg["txarea"] = out["TxLoopArea"]
    txp = collectNumData(out, "TxLoopPoint")
    cfg["tx"], cfg["ty"] = txp[:, 0], txp[:, 1]
    bla = collectNumData(out, "WaveformLMPoint", num=2)
    cfg["tL"], cfg["vL"] = bla[:, 0], bla[:, 1]
    bla = collectNumData(out, "WaveformHMPoint", num=2)
    cfg["tH"], cfg["vH"] = bla[:, 0], bla[:, 1]
    cfg["timeL"] = collectNumData(out, "GateTimeLM")[:, 0]
    cfg["timeH"] = collectNumData(out, "GateTimeHM")[:, 0]
    return cfg

def readSettings1(filename="sTEM.gex"):
    with open(filename) as fid:
        lines = fid.readlines()

    cfg = {}
    cfg["tL"], cfg["vL"] = np.genfromtxt(lines[20:60], usecols=[1, 2], unpack=True)
    cfg["tH"], cfg["vH"] = np.genfromtxt(lines[61:109], usecols=[1, 2], unpack=True)
    cfg["timeL"] = np.genfromtxt(lines[111:117], usecols=[1])
    cfg["timeH"] = np.genfromtxt(lines[119:141], usecols=[1])
    cfg["tx"], cfg["ty"] = np.genfromtxt(lines[15:19], usecols=[1, 2], unpack=True)
    cfg["rxpos"] = np.genfromtxt(lines[7:8], usecols=[1, 2, 3])
    cfg["txarea"] = np.genfromtxt(lines[14:15], usecols=[1])
    return cfg


def bandpass(inp, p_dict):
    """Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)."""
    cutofffreq = 1e8  # Determined empirically for TEM-FAST
    h = (1 + 1j*p_dict["freq"]/cutofffreq)**-1
    h *= (1 + 1j*p_dict["freq"]/3e5)**-1
    p_dict["EM"] *= h[:, None]


class sTEMBlockModelling(pg.frameworks.Block1DModelling):

    def __init__(self, **kwargs):
        cfg = kwargs.pop("cfg", {})
        super().__init__(**kwargs)
        if isinstance(cfg, str):
            cfg = readSettings(cfg)

        self.signalL = {'nodes': cfg["tL"], 'amplitudes': cfg["vL"], 'signal': 1}
        self.signalH = {'nodes': cfg["tH"], 'amplitudes': cfg["vH"], 'signal': 1}
        self.timeL = cfg["timeL"]
        self.timeH = cfg["timeH"]

        self.kw = dict(
            src=[[cfg["tx"][-1], *cfg["tx"]], # x1
            [*cfg["tx"], cfg["tx"][0]], # x2
            [cfg["ty"][-1], *cfg["ty"]], # y1
            [*cfg["ty"], cfg["ty"][0]], # y2
            0, 0],
            strength=1/cfg["txarea"],
            verb=0,
            rec = np.concatenate([cfg["rxpos"], [0, 90]]),       # Receiver at the origin, vertical.
            mrec="b",                   # Receiver: dB/dt
            srcpts=3,                   # Approx. the finite dip. with 3 points.
            ftarg={"dlf": "key_81_2009"},  # Shorter, faster filters.
            htarg={"dlf": "key_101_2009", "pts_per_dec": -1},
            bandpass={"func": bandpass}
            )

    @property
    def t(self):
        return np.concatenate([self.timeL, self.timeH])

    def response(self, model):
        """Return model response."""
        thk = model[:self.nLayers-1]
        res = model[self.nLayers-1:]
        outL = empymod.model.bipole(
            depth=np.concatenate([[0], np.cumsum(np.atleast_1d(thk))]), # Depth-model.
            res=np.concatenate([[2e14], np.atleast_1d(res)]),      # Resistivity model.
            signal=self.signalL,
            freqtime=self.timeL,      # Wanted times.
            **self.kw)
        outH = empymod.model.bipole(
            depth=np.concatenate([[0], np.cumsum(np.atleast_1d(thk))]), # Depth-model.
            res=np.concatenate([[2e14], np.atleast_1d(res)]),      # Resistivity model.
            signal=self.signalH,
            freqtime=self.timeH,      # Wanted times.
            **self.kw)
        return np.concatenate([outL.sum(axis=1), outH.sum(axis=1)])


class sTEMRhoModelling(pg.frameworks.MeshModelling):

    def __init__(self, thk, **kwargs):
        self.thk = thk
        cfg = kwargs.pop("cfg", "sTEM.gex")
        self.mesh_ = pg.meshtools.createMesh1D(len(thk)+1)
        super().__init__(mesh=self.mesh_)
        if isinstance(cfg, str):
            cfg = readSettings(cfg)
        # t = cfg["tL"]
        # v = cfg["vL"]
        # self.signalL = {"nodes": t[t >= -t[-1]*100], "amplitudes": v[t >= -t[-1]*100]}
        # t = cfg["tH"]
        # v = cfg["vH"]
        # self.signalH = {"nodes": t[t >= -t[-1]*100], "amplitudes": v[t >= -t[-1]*100]}
        self.signalL = {'nodes': cfg["tL"], 'amplitudes': cfg["vL"], 'signal': 1}
        self.signalH = {'nodes': cfg["tH"], 'amplitudes': cfg["vH"], 'signal': 1}
        self.timeL = cfg["timeL"]
        self.timeH = cfg["timeH"]

        self.kw = dict(
            src=[[cfg["tx"][-1], *cfg["tx"]], # x1
            [*cfg["tx"], cfg["tx"][0]], # x2
            [cfg["ty"][-1], *cfg["ty"]], # y1
            [*cfg["ty"], cfg["ty"][0]], # y2
            0, 0],
            strength=1/cfg["txarea"],
            verb=0,
            depth = np.concatenate([[0], np.cumsum(np.atleast_1d(thk))]),
            rec = np.concatenate([cfg["rxpos"], [0, 90]]), # Receiver at the origin, vertical.
            mrec="b",                   # Receiver: dB/dt
            srcpts=3,                   # Approx. the finite dip. with 3 points.
            ftarg={"dlf": "key_81_2009"},  # Shorter, faster filters.
            htarg={"dlf": "key_101_2009", "pts_per_dec": -1},
            bandpass={"func": bandpass}
            )

    @property
    def t(self):
        return np.concatenate([self.timeL, self.timeH])

    def response(self, model):
        """Return model response."""
        return np.concatenate([
            empymod.model.bipole(
                res=np.concatenate([[2e14], model]),
                signal=self.signalL,
                freqtime=self.timeL,      # Wanted times.
                **self.kw).sum(axis=1),
            empymod.model.bipole(
                res=np.concatenate([[2e14], model]),
                signal=self.signalH,
                freqtime=self.timeH,      # Wanted times.
                **self.kw).sum(axis=1)])

    def createStartVector(self, data):
        return pg.Vector(len(self.thk)+1, 250.)

# %%
if __name__ == "__main__":
    f = sTEMRhoModelling(thk=np.arange(2, 28, 2), cfg="sTEM.gex")
    rho = pg.Vector(len(f.thk)+1, 100.)
    print(f(rho))
