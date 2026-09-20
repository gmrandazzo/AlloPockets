"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Original Author: Francho Nerín Fonz <fnerin@bioacademy.gr>
Refactoring & PyRosetta Integration: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
License: GNU General Public License v3.0 (GPL-3.0)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
from functools import cached_property, cache

# ## Biopython

# In[11]:


from Bio import PDB  # conda Biopython 1.84

# mkdssp executable downloaded from https://github.com/PDB-REDO/dssp/releases/tag/v4.4.0
# dssp can be installed in env avoiding compatibility issues with pyroseta with:
## wget https://conda.anaconda.org/conda-forge/linux-64/dssp-4.4.8-h629725b_0.conda
## conda install dssp-4.4.8-h629725b_0.conda
import shutil
from .aa_scales import get_residues_scales_df

try:
    import melodia_py as melodia
except ImportError:
    melodia = None
from Bio.SeqUtils import seq1

# In[12]:


class BiopythonF:
    dssp_path = shutil.which("mkdssp") or "mkdssp-4.4.0-linux-x64"

    def __init__(self, cif):
        self._cif = cif

    features = ["exposureCB", "exposureCN", "residue_depth"]

    @cached_property
    def struc(self):
        return PDB.MMCIFParser().get_structure(self._cif.entry_id.upper(), self._cif.filename)

    @property
    def model(self):
        return self.struc[0]

    def _get_res_d(self, r):
        return {
            "auth_asym_id": r.full_id[2],
            "auth_seq_id": str(r.full_id[3][1]),
            "pdbx_PDB_ins_code": r.full_id[3][2].replace(" ", "") or "?",
        }

    def _get_res_df(self, r):
        return pd.DataFrame([self._get_res_d(r)], dtype=str)

    def _process_property_list(self, property_list, ps):
        return pd.concat(
            [
                pd.concat([self._get_res_df(r), pd.DataFrame([{p: r.xtra[p] for p in ps}])], axis=1)
                for i in property_list
                for r in [
                    i[0],
                ]
            ]
        )

    def _exposureCB(self):
        return PDB.HSExposure.HSExposureCB(self.model)

    def exposureCB(self):
        return self._process_property_list(
            self._exposureCB().property_list, ["EXP_HSE_B_U", "EXP_HSE_B_D"]
        )

    def _exposureCN(self):
        return PDB.HSExposure.ExposureCN(self.model)

    def exposureCN(self):
        return self._process_property_list(self._exposureCN().property_list, ["EXP_CN"])

    def _residue_depth(self):
        try:
            return PDB.ResidueDepth(self.model)
        except Exception:
            return None

    def residue_depth(self):
        rd = self._residue_depth()
        if rd is not None and hasattr(rd, "property_list"):
            return self._process_property_list(rd.property_list, ["EXP_RD", "EXP_RD_CA"])
        return pd.DataFrame(
            [
                {
                    "auth_asym_id": r.full_id[2],
                    "auth_seq_id": str(r.full_id[3][1]),
                    "pdbx_PDB_ins_code": r.full_id[3][2].replace(" ", "") or "?",
                    "EXP_RD": float("nan"),
                    "EXP_RD_CA": float("nan"),
                }
                for r in self.model.get_residues()
                if r.id[0] == " "
            ]
        )

    # Might fail because some residues are renamed during structure fixing
    def _dssp(self):
        extra = pd.DataFrame(self._cif.origcif.data["_pdbx_poly_seq_scheme"], dtype="str")
        extra = (
            extra.merge(
                self._cif.residues,
                left_on=["asym_id", "entity_id", "seq_id"],
                right_on=["label_asym_id", "label_entity_id", "label_seq_id"],
            )[extra.columns]
            .drop_duplicates()
            .to_dict(orient="list")
        )

        with self._cif._extended_temp_ciff({"_pdbx_poly_seq_scheme": extra}) as f:
            struc = PDB.MMCIFParser().get_structure(self._cif.entry_id.upper(), f.name)
            model = struc[0]
            dssp = PDB.DSSP(model, f.name, dssp=self.dssp_path)
        return dssp

    def dssp(self):
        return pd.DataFrame(
            [
                {
                    "auth_asym_id": k[0],
                    "auth_seq_id": str(k[1][1]),
                    "pdbx_PDB_ins_code": k[1][2].replace(" ", "") or "?",
                    **dict(
                        zip(
                            "dssp index, amino acid, secondary structure, relative ASA, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy".split(
                                ", "
                            ),
                            v,
                        )
                    ),
                }
                for k, v in self._dssp().property_dict.items()
            ]
        ).drop(["dssp index", "amino acid"], axis=1)

    def _melodia(self):
        if melodia is None:
            raise ImportError(
                "melodia-py is missing or failed to import (often due to missing 'nglview' or 'numba' dependencies). Please run 'pip install melodia-py nglview numba'."
            )
        records = []
        struc = self.struc
        geom = melodia.geometry_dict_from_structure(struc)
        for chain_key, c in geom.items():
            chain_id = chain_key.split(":")[-1]
            if chain_id not in struc[0]:
                continue
            chain = struc[0][chain_id]
            bio_residues = [r for r in chain.get_residues() if r.id[0] == " "]
            for res_idx, r in c.residues.items():
                if res_idx < len(bio_residues):
                    bio_res = bio_residues[res_idx]
                    records.append(
                        {
                            "auth_asym_id": chain_id,
                            "auth_seq_id": str(bio_res.id[1]),
                            "auth_comp_id": r.name,
                            "pdbx_PDB_ins_code": bio_res.id[2].replace(" ", "") or "?",
                            "curvature": r.curvature,
                            "torsion": r.torsion,
                            "arc_len": r.arc_len,
                            "writhing": r.writhing,
                            "phi": r.phi,
                            "psi": r.psi,
                        }
                    )
        return pd.DataFrame(records)

    def melodia(self):
        df = self._melodia()

        # Add melodia_propensity
        ptable = melodia.PropensityTable()
        df["propensity"] = [
            ptable.get_score(
                target="A" if r["auth_comp_id"] != "ALA" else "G",
                residue=seq1(r["auth_comp_id"]),
                phi=r["phi"],
                psi=r["psi"],
            )
            for i, r in df.iterrows()
        ]

        return df.drop(columns="auth_comp_id")


# ### DSSP

# In[13]:


class DSSPF:
    def __init__(self, cif):
        self._cif = cif
        self.dssp = BiopythonF(self._cif).dssp

    features = ["dssp"]


# ### Melodia_py

# In[14]:


class MelodiaF:
    def __init__(self, cif):
        self._cif = cif
        self.melodia = BiopythonF(self._cif).melodia

    features = ["melodia"]


# ## Graphein

# In[15]:


try:
    from graphein import protein as graphein
except ImportError:
    graphein = None


# In[16]:


class GrapheinF:
    def __init__(self, cif):
        self._cif = cif

    features = ["graphein"]

    def graphein(self):
        return get_residues_scales_df(self._cif.residues)


# ## FreeSASA

# In[17]:


import subprocess, json

# In[18]:


class FreeSASAF:
    def __init__(self, cif):
        self._cif = cif

    features = ["freesasa"]

    def _freesasa(self):
        try:
            result = subprocess.run(
                [f"freesasa", "--depth=residue", "--cif", "--format=json", self._cif.filename],
                capture_output=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "freesasa is not installed. Please install it (e.g. 'brew install freesasa' on Mac or 'sudo apt-get install freesasa' on Linux)."
            )
        return json.loads(result.stdout.decode().strip())

    def freesasa(self):
        resdf = pd.DataFrame(
            self._cif.residues[["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]]
        )
        resdf["freesasaid"] = [
            f"{num}{ic.replace('?','')}"
            for num, ic in resdf[["auth_seq_id", "pdbx_PDB_ins_code"]].values
        ]
        df = pd.DataFrame(
            [
                {
                    "auth_asym_id": chain["label"],
                    "freesasaid": str(res["number"]),
                    **{
                        f"{a}_{t}": tv
                        for a in ["area", "relative-area"]
                        for t, tv in res[a].items()
                    },
                }
                for chain in self._freesasa()["results"][0]["structure"][0]["chains"]
                for res in chain["residues"]
            ]
        )
        return resdf.merge(df).drop(columns="freesasaid")


## PyRosetta

# In[21]:s


try:
    from pyrosetta import init, pose_from_pdb, get_score_function
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
    from pyrosetta.rosetta.utility import vector1_bool
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta import Pose

    HAS_PYROSETTA = True
except ImportError:
    HAS_PYROSETTA = False

_pyrosetta_initialized = False


def _init_pyrosetta():
    global _pyrosetta_initialized
    if HAS_PYROSETTA and not _pyrosetta_initialized:
        init("-mute core.pack basic core.scoring -ignore_zero_occupancy false")
        _pyrosetta_initialized = True


# In[22]:


class PyRosettaF:
    def __init__(self, cif):
        self._cif = cif
        _init_pyrosetta()
        if not HAS_PYROSETTA:
            print(
                "WARNING: PyRosetta is not installed. Skipping ddG calculations (features will be NaN)."
            )

    features = ["ddG"]

    @cached_property
    def _pose(self):
        if not HAS_PYROSETTA:
            return None
        return pose_from_pdb(self._cif.filename)

    def _mutate_and_repack(self, pose, resnum, target_aa, sfxn):
        mut_pose = Pose()
        mut_pose.assign(pose)

        # Mutate
        mut_mover = MutateResidue(resnum, target_aa)
        mut_mover.apply(mut_pose)

        # Repack 8A radius
        task = TaskFactory.create_packer_task(mut_pose)
        task.restrict_to_repacking()
        # Prevent repacking of residues further than 8A
        center = mut_pose.residue(resnum).xyz("CA")
        for i in range(1, mut_pose.total_residue() + 1):
            if mut_pose.residue(i).xyz("CA").distance(center) > 8.0:
                task.nonconst_residue_task(i).prevent_repacking()

        packer = PackRotamersMover(sfxn, task)
        packer.apply(mut_pose)
        return mut_pose

    def _ddG(self):
        if not HAS_PYROSETTA:
            import numpy as np

            # Return NaNs if PyRosetta is missing
            for _, rec in self._cif.residues.iterrows():
                yield {
                    "auth_asym_id": rec["auth_asym_id"],
                    "auth_seq_id": str(rec["auth_seq_id"]),
                    "pdbx_PDB_ins_code": rec.get("pdbx_PDB_ins_code", "?"),
                    "ddG": np.nan,
                }
            return

        sfxn = get_score_function(True)
        pose = self._pose
        pdbinfo = pose.pdb_info()

        for i, res in enumerate(pose.residues, 1):
            aa1 = res.name1()
            resnum = res.seqpos()
            aa3 = res.name3()
            target_aa = "ALA" if aa1 != "A" else "GLY"

            # Mutate to target AA and repack
            mutated_pose = self._mutate_and_repack(pose, resnum, target_aa, sfxn)
            score_mut = sfxn.score(mutated_pose)

            # Re-repack native AA to maintain fair baseline
            native_pose = self._mutate_and_repack(pose, resnum, aa3, sfxn)
            score_nat = sfxn.score(native_pose)

            ddG = score_mut - score_nat

            yield {
                "auth_asym_id": pdbinfo.chain(i),
                "auth_seq_id": str(pdbinfo.number(i)),
                "pdbx_PDB_ins_code": pdbinfo.icode(i).replace(" ", "") or "?",
                "ddG": ddG,
            }

    def ddG(self):
        return pd.DataFrame(self._ddG())


## ProDy

# In[19]:


try:
    import prody

    prody.confProDy(verbosity="none")
except ImportError:
    prody = None

import numpy as np

# In[20]:


class ProDyF:
    def __init__(self, cif):
        self._cif = cif
        if prody is None:
            raise ImportError("prody is required for ProDyF. Please install ProDy.")

    features = [
        "prs",
        "mechstiff",
        "rmsf",
        "essa",
    ]

    @cached_property
    def _atoms(self):
        return prody.parseMMCIF(self._cif.filename)

    @property
    def _cas(self):
        return self._atoms.select("name CA")

    @cached_property
    def _res_df(self):
        return pd.DataFrame(
            (
                {
                    "auth_asym_id": res.getChid(),
                    "auth_seq_id": str(res.getResnum()),
                    "pdbx_PDB_ins_code": res.getIcode() or "?",
                }
                for res in self._atoms.iterResidues()
            )
        )

    def _get_df(self, colname, col, df=None):
        if df is None:
            df = pd.DataFrame(self._res_df)
        df[colname] = col
        return df

    @cache
    def _anm(
        self,
        n_modes=50,
        **kwargs,
    ):
        anm = prody.ANM()
        anm.buildHessian(self._cas, **kwargs)
        anm.calcModes(n_modes=n_modes, **kwargs)
        return anm

    def _prs(self, **anm_kwargs):
        return prody.calcPerturbResponse(
            self._anm(**anm_kwargs), turbo=True
        )  # turbo false doesn't work in prody 2.4.1

    def prs(self, **anm_kwargs):
        _, effectiveness, sensitivity = self._prs(**anm_kwargs)
        df = self._get_df("prs_effectiveness", effectiveness)
        df = self._get_df("prs_sensitivity", sensitivity, df=df)
        return df

    def _mechstiff(self, **anm_kwargs):
        return prody.calcMechStiff(self._anm(**anm_kwargs), self._cas)

    def mechstiff(self, **anm_kwargs):
        meanstiff = np.mean(
            self._mechstiff(**anm_kwargs), axis=0
        )  # from showMeanMechStiff function
        return self._get_df("mechstiff", meanstiff)

    def _rmsflucts(self, n_modes=20, **anm_kwargs):
        anm = self._anm(**anm_kwargs)
        return prody.calcRMSFlucts(anm if n_modes is None else anm[:n_modes])

    def rmsf(self, n_modes=20, **anm_kwargs):
        return self._get_df("rmsf", self._rmsflucts(n_modes=n_modes, **anm_kwargs))

    def _essa(
        self,
        enm="gnm",
        lowmem=True,
        n_modes=10,
        **kwargs,
    ):
        essa = prody.ESSA()
        essa.setSystem(self._atoms, lowmem=lowmem, **kwargs)
        essa.scanResidues(enm=enm, n_modes=n_modes)
        return essa

    def essa(self, max_residues=1200, **kwargs):
        if len(self._cas) > max_residues:
            # Skip ESSA on large multimers to prevent O(N^3) pairwise distance bottleneck
            return self._get_df("essa", np.full(len(self._res_df), np.nan))
        try:
            essa = self._essa(**kwargs).getESSAZscores()
            return self._get_df("essa", essa)
        except Exception:
            return self._get_df("essa", np.full(len(self._res_df), np.nan))


# ## HHBlits

# In[23]:


try:
    from moleculekit.tools.hhblitsprofile import getSequenceProfile
except ImportError:
    getSequenceProfile = None


# In[24]:


class HHBlitsF:

    uniref_path = "."

    def __init__(self, cif):
        self._cif = cif

    features = ["hhblits"]

    def _hhblits(self, seq, entity_id, ncpu):
        if getSequenceProfile is None:
            raise ImportError(
                "moleculekit is required for HHBlitsF._hhblits. Please install moleculekit."
            )
        return getSequenceProfile(
            seq, hhblits="hhblits", hhblitsdb=self.uniref_path, ncpu=ncpu, niter=4
        )

    def hhblits(self, ncpu=10):
        dfs = []
        res = self._cif.residues
        ents = pd.DataFrame(self._cif.cif.data["_entity_poly"], dtype=str)
        for entity_id, entity_res in res.groupby("label_entity_id"):
            seq = (
                ents.query(f"entity_id == '{entity_id}'")["pdbx_seq_one_letter_code_can"]
                .item()
                .replace("\n", "")
            )
            df, _ = self._hhblits(seq, ncpu=ncpu, entity_id=entity_id)
            for asym_id in entity_res.label_asym_id.unique():
                dfs.append(
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "label_asym_id": [asym_id] * len(seq),
                                    "label_seq_id": range(1, len(seq) + 1),
                                },
                                dtype=str,
                            ),
                            df.iloc[1:].drop(columns="seq").reset_index(drop=True),
                        ],
                        axis=1,
                    )
                )
        hhblits = pd.concat(dfs)
        return res.merge(hhblits, on=["label_asym_id", "label_seq_id"])[hhblits.columns]


## Transfer entropy

# In[25]:


import biotite.structure.io.pdbx
from biotite import structure as biotite_structure

# biotite 1.0.1; other versions may change the shape of the atom_array?


# In[26]:


class Biotite_struc:
    def __init__(self, cif):
        self._cif = cif

    @property
    def struc(self):
        return biotite_structure.io.pdbx.get_structure(
            biotite_structure.io.pdbx.CIFFile.read(self._cif.filename), model=1
        )

    @property
    def _atom_df(self):
        atom_array = self.struc
        return pd.DataFrame(
            {
                "auth_asym_id": atom_array.chain_id,
                "auth_seq_id": atom_array.res_id,
                "auth_atom_id": atom_array.atom_name,
                "type_symbol": atom_array.element,
                "Cartn_x": atom_array.coord[:, 0],
                "Cartn_y": atom_array.coord[:, 1],
                "Cartn_z": atom_array.coord[:, 2],
                "pdbx_PDB_ins_code": (ic or "?" for ic in atom_array.ins_code),
            },
            dtype=str,
        )

    @property
    def _res_df(self):
        return self._atom_df[["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]].drop_duplicates()


# In[27]:


# from AllosES: https://github.com/ChunhuaLab/AllosES/blob/main/AllosES/utils.py
import math
from threadpoolctl import threadpool_limits

# numpy is old (1.26) due to ProDy requirements; newer might be compatible with numba...


# In[28]:


class TransferEntropyF:
    def __init__(self, cif):
        self._cif = cif
        self._biotite = Biotite_struc(self._cif)

    features = ["transfer_entropy"]

    @staticmethod
    def _distance(coordinate_matrix):
        size, _ = np.shape(coordinate_matrix)
        dis = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                if j == i:
                    continue
                else:
                    dis[i, j] = math.sqrt(
                        (coordinate_matrix[i, 0] - coordinate_matrix[j, 0]) ** 2
                        + (coordinate_matrix[i, 1] - coordinate_matrix[j, 1]) ** 2
                        + (coordinate_matrix[i, 2] - coordinate_matrix[j, 2]) ** 2
                    )
        return dis

    def _kirchhoff(self, coordinate_matrix, cutoff):
        size, _ = np.shape(coordinate_matrix)
        Kirchhoff_matrix = np.zeros([size, size])
        dis = TransferEntropyF._distance(coordinate_matrix)
        for i in range(size):
            for j in range(size):
                if j == i:
                    continue
                elif j != i:  # Non-diagonal
                    if dis[i, j] <= cutoff:
                        Kirchhoff_matrix[i, j] = -1
                    else:
                        Kirchhoff_matrix[i, j] = 0
            Kirchhoff_matrix[i, i] = -1 * sum(Kirchhoff_matrix[i, :])
        return Kirchhoff_matrix

    def _GNM(self, coordinate, N, cutoff):
        Kirchhoff = self._kirchhoff(coordinate, cutoff)
        [Vectors, Values, VectorsT1] = np.linalg.svd(Kirchhoff)
        sorted_indices = np.argsort(Values)
        Values = Values[sorted_indices[::1]]
        Vectors = Vectors[:, sorted_indices[::1]]
        InvKirchhoff = (Vectors) * (np.linalg.pinv(np.diag(Values))) * (Vectors.T)
        CellAij = {}
        for k in range(0, N):
            if (1 / Values[k]) < 1000:
                CellAij[k] = Vectors[:, k] * (np.array([Vectors[:, k]]).T) / Values[k]
            else:
                CellAij[k] = np.zeros([N, N])
        return InvKirchhoff, CellAij, N, Values

    def _Transfer_entropy(self, coordinate, N, cutoff, tau):
        with threadpool_limits(limits=2):
            Kirchhoff = self._kirchhoff(coordinate, cutoff)
            Vectors, Values, _ = np.linalg.svd(Kirchhoff)
            sorted_indices = np.argsort(Values)
            Values = Values[sorted_indices]
            Vectors = Vectors[:, sorted_indices]

            inv_vals = np.where(1 / Values < 1000, 1 / Values, 0.0)
            exp_eig = np.exp(-Values * tau)

            sum_d = (Vectors * inv_vals) @ Vectors.T
            sum_b = (Vectors * (inv_vals * exp_eig)) @ Vectors.T
            sum_c = np.diag(sum_d)
            sum_a = np.diag(sum_b)

            sum_c_j = sum_c[None, :]
            sum_c_i = sum_c[:, None]
            sum_a_j = sum_a[None, :]

            a = sum_c_j**2 - sum_a_j**2
            b = sum_c_i * (sum_c_j**2)
            c = 2 * sum_d * sum_a_j * sum_b
            d = -((sum_b**2 + sum_d**2) * sum_c_j) - ((sum_a_j**2) * sum_c_i)
            f = sum_c_j
            g = sum_c_i * sum_c_j - sum_d**2

            with np.errstate(divide="ignore", invalid="ignore"):
                TE = (
                    0.5 * np.log(a.astype(complex))
                    - 0.5 * np.log((b + c + d).astype(complex))
                    - 0.5 * np.log(f.astype(complex))
                    + 0.5 * np.log(g.astype(complex))
                )
            np.fill_diagonal(TE, 0)
            TE[TE < 0] = 0
            netTE = TE - TE.T
            Difference = np.real((netTE).sum(axis=1))
            max_val = np.max(np.abs(Difference))
            norm_difference = Difference / max_val if max_val > 0 else Difference
            return norm_difference

    def _transfer_entropy(self, cutoff, tau):
        atom_array = self._biotite.struc
        cas = atom_array[atom_array.atom_name == "CA"]
        N = len(cas)
        coordinate = cas.coord

        return self._Transfer_entropy(coordinate, N, cutoff, tau)

    def transfer_entropy(self, cutoff=7, tau=5):
        te = self._transfer_entropy(cutoff, tau)
        df = self._biotite._res_df
        assert len(te) == len(df)
        df["TE"] = te
        return df


FClasses = [
    GrapheinF,
    FreeSASAF,
    DSSPF,
    MelodiaF,
    BiopythonF,
    PyRosettaF,
    ProDyF,
    TransferEntropyF,
    HHBlitsF,
]
