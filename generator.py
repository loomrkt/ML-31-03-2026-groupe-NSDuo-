from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple, Dict
import pandas as pd

Board = Tuple[str, ...]


WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def check_winner(board: Board) -> str | None:
    """Retourne 'X' si X a gagné, 'O' si O a gagné, sinon None."""
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board: Board) -> bool:
    """Vrai si le plateau est plein."""
    return all(cell != " " for cell in board)


def is_valid_state(board: Board) -> bool:
    """
    Vérifie si l'état est valide dans une vraie partie :
    - X commence toujours
    - nombre de X = nombre de O ou nombre de X = nombre de O + 1
    - pas de double victoire incohérente
    """
    x_count = board.count("X")
    o_count = board.count("O")

    if not (x_count == o_count or x_count == o_count + 1):
        return False

    x_wins = False
    o_wins = False

    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] == "X":
            x_wins = True
        if board[a] == board[b] == board[c] == "O":
            o_wins = True

    # Les deux ne peuvent pas gagner en même temps
    if x_wins and o_wins:
        return False

    # Si X a gagné, il doit avoir joué un coup de plus que O
    if x_wins and x_count != o_count + 1:
        return False

    # Si O a gagné, il doit avoir joué autant de coups que X
    if o_wins and x_count != o_count:
        return False

    return True


def get_Succ(board: Board) -> str:
    """
    Détermine le joueur courant.
    X commence. Donc :
    - si X et O ont joué autant => à X
    - sinon => à O
    """
    x_count = board.count("X")
    o_count = board.count("O")
    return "X" if x_count == o_count else "O"


@lru_cache(maxsize=None)
def minimax_outcome(board: Board, player: str) -> str:
    """
    Retourne l'issue théorique parfaite depuis cet état :
    - 'X' si X peut forcer la victoire
    - 'O' si O peut forcer la victoire
    - 'DRAW' si match nul parfait
    """
    winner = check_winner(board)
    if winner is not None:
        return winner

    if is_full(board):
        return "DRAW"

    empty_cells = [i for i, cell in enumerate(board) if cell == " "]

    outcomes = []
    for idx in empty_cells:
        new_board = list(board)
        new_board[idx] = player
        new_board_t = tuple(new_board)

        next_p = "O" if player == "X" else "X"
        outcome = minimax_outcome(new_board_t, next_p)
        outcomes.append(outcome)

    # Le joueur courant choisit le meilleur résultat pour lui
    if player == "X":
        if "X" in outcomes:
            return "X"
        if "DRAW" in outcomes:
            return "DRAW"
        return "O"
    else:
        if "O" in outcomes:
            return "O"
        if "DRAW" in outcomes:
            return "DRAW"
        return "X"


def encode_board(board: Board) -> Dict[str, int]:
    """
    Transforme un plateau en 18 features :
    c0_x, c0_o, ..., c8_x, c8_o
    """
    features: Dict[str, int] = {}
    for i, cell in enumerate(board):
        features[f"c{i}_x"] = 1 if cell == "X" else 0
        features[f"c{i}_o"] = 1 if cell == "O" else 0
    return features


def generate_all_boards() -> List[Board]:
    """
    Génère tous les plateaux possibles (3^9), puis filtre les états valides.
    """
    symbols = [" ", "X", "O"]
    valid_boards: List[Board] = []

    def backtrack(pos: int, current: List[str]) -> None:
        if pos == 9:
            board = tuple(current)
            if is_valid_state(board):
                valid_boards.append(board)
            return

        for s in symbols:
            current.append(s)
            backtrack(pos + 1, current)
            current.pop()

    backtrack(0, [])
    return valid_boards


def build_dataset() -> pd.DataFrame:
    rows = []
    boards = generate_all_boards()

    for board in boards:
        # ignorer les états terminaux
        if check_winner(board) is not None or is_full(board):
            continue

        # garder uniquement les états où c'est au tour de X
        if get_Succ(board) != "X":
            continue

        outcome = minimax_outcome(board, "X")

        row = encode_board(board)
        row["x_wins"] = 1 if outcome == "X" else 0
        row["is_draw"] = 1 if outcome == "DRAW" else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # ordre exact des colonnes
    ordered_cols = []
    for i in range(9):
        ordered_cols.append(f"c{i}_x")
        ordered_cols.append(f"c{i}_o")
    ordered_cols += ["x_wins", "is_draw"]

    df = df[ordered_cols]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset()

    os.makedirs("ressources", exist_ok=True)
    output_path = os.path.join("ressources", "dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset généré : {output_path}")
    print(f"Nombre de lignes : {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()