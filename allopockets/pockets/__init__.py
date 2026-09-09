"""
AlloPockets - Allosteric Pocket Prediction, Pathway Tracing & 3D Debugging.

Original Author: Francho Nerín Fonz <fnerin@bioacademy.gr>
Refactoring & Packaging: Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
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

Pocket detection and geometric analysis module.
"""

from allopockets.pockets.pocket import Pocket, get_pockets_info
from allopockets.pockets.fpocket import run_fpocket, get_pocket_atoms, is_fpocket_available

__all__ = ["Pocket", "get_pockets_info", "run_fpocket", "get_pocket_atoms", "is_fpocket_available"]
