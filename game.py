from __future__ import annotations

import math
from typing import List, Optional

import customtkinter as ctk
import joblib


# =========================================================
# CONFIGURATION GLOBALE
# =========================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# =========================================================
# CHARGEMENT DES MODELES
# =========================================================
MODEL_X = joblib.load("ressources/model_xwins.pkl")
MODEL_D = joblib.load("ressources/model_draw.pkl")


# =========================================================
# CONSTANTES JEU
# =========================================================
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


# =========================================================
# LOGIQUE MORPION
# =========================================================
def check_winner(board: List[str]) -> Optional[str]:
    """Retourne 'X', 'O' ou None."""
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board: List[str]) -> bool:
    """Retourne True si le plateau est plein."""
    return all(cell != " " for cell in board)


def available_moves(board: List[str]) -> List[int]:
    """Liste les cases vides."""
    return [i for i, cell in enumerate(board) if cell == " "]


def make_move(board: List[str], move: int, player: str) -> List[str]:
    """Retourne un nouveau plateau après un coup."""
    new_board = board.copy()
    new_board[move] = player
    return new_board


def next_player(board: List[str]) -> str:
    """
    Détermine à qui est le tour.
    X commence toujours.
    Si X et O ont joué autant => tour de X
    Sinon => tour de O
    """
    x_count = board.count("X")
    o_count = board.count("O")
    return "X" if x_count == o_count else "O"


def swap_board_perspective(board: List[str]) -> List[str]:
    """
    Inverse X et O dans le plateau.
    Utile car les modèles ont été entraînés seulement
    sur les états où c'est au tour de X.
    """
    swapped = []
    for cell in board:
        if cell == "X":
            swapped.append("O")
        elif cell == "O":
            swapped.append("X")
        else:
            swapped.append(" ")
    return swapped


def encode_board(board: List[str]) -> List[int]:
    """
    Encode le plateau en 18 features :
    c0_x, c0_o, ..., c8_x, c8_o
    """
    features: List[int] = []
    for cell in board:
        features.append(1 if cell == "X" else 0)
        features.append(1 if cell == "O" else 0)
    return features


def evaluate_with_ml(board: List[str]) -> float:
    """
    Retourne un score du point de vue de X.
    +1 = très bon pour X
    -1 = très bon pour O
     0 = neutre

    IMPORTANT :
    Les modèles ont été entraînés sur les états où c'est au tour de X.
    Donc si ce n'est pas le tour de X, on inverse X/O,
    on prédit, puis on réinverse le score.
    """
    winner = check_winner(board)
    if winner == "X":
        return 1.0
    if winner == "O":
        return -1.0
    if is_full(board):
        return 0.0

    current = next_player(board)

    # Cas normal : c'est bien à X de jouer
    if current == "X":
        features = encode_board(board)
        p_xwins = MODEL_X.predict_proba([features])[0][1]
        p_draw = MODEL_D.predict_proba([features])[0][1]
        p_owins = max(0.0, 1.0 - p_xwins - p_draw)
        return float(p_xwins - p_owins)

    # Cas où c'est à O de jouer :
    # on inverse le plateau pour ramener ça à "tour de X"
    mirrored = swap_board_perspective(board)
    features = encode_board(mirrored)
    p_xwins = MODEL_X.predict_proba([features])[0][1]
    p_draw = MODEL_D.predict_proba([features])[0][1]
    p_owins = max(0.0, 1.0 - p_xwins - p_draw)

    mirrored_score = p_xwins - p_owins

    # On revient au vrai point de vue de X
    return float(-mirrored_score)


def find_immediate_winning_move(board: List[str], player: str) -> Optional[int]:
    """
    Cherche un coup gagnant immédiat pour le joueur donné.
    Retourne l'index du coup s'il existe.
    """
    for move in available_moves(board):
        test_board = make_move(board, move, player)
        if check_winner(test_board) == player:
            return move
    return None


# =========================================================
# IA ML PURE
# =========================================================
def best_move_ml(board: List[str], ai_player: str = "O") -> int:
    """
    IA ML :
    1. gagne si possible
    2. bloque l'adversaire si nécessaire
    3. sinon choisit selon l'évaluation ML
    """
    moves = available_moves(board)
    if not moves:
        return -1

    human_player = "O" if ai_player == "X" else "X"

    # 1) gagner immédiatement
    win_move = find_immediate_winning_move(board, ai_player)
    if win_move is not None:
        return win_move

    # 2) bloquer l'adversaire
    block_move = find_immediate_winning_move(board, human_player)
    if block_move is not None:
        return block_move

    # 3) évaluation ML
    best_move = moves[0]

    if ai_player == "X":
        best_score = -math.inf
        for move in moves:
            new_board = make_move(board, move, "X")
            score = evaluate_with_ml(new_board)
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = math.inf
        for move in moves:
            new_board = make_move(board, move, "O")
            score = evaluate_with_ml(new_board)
            if score < best_score:
                best_score = score
                best_move = move

    return best_move


# =========================================================
# IA HYBRIDE : MINIMAX + ML
# =========================================================
def minimax_hybrid(
    board: List[str],
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> float:
    """
    maximizing=True  => tour de X
    maximizing=False => tour de O
    """
    winner = check_winner(board)
    if winner == "X":
        return 1.0
    if winner == "O":
        return -1.0
    if is_full(board):
        return 0.0

    if depth == 0:
        return evaluate_with_ml(board)

    moves = available_moves(board)

    if maximizing:
        value = -math.inf
        for move in moves:
            child = make_move(board, move, "X")
            value = max(value, minimax_hybrid(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = math.inf
    for move in moves:
        child = make_move(board, move, "O")
        value = min(value, minimax_hybrid(child, depth - 1, alpha, beta, True))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def best_move_hybrid(board: List[str], ai_player: str = "O", depth: int = 3) -> int:
    """
    IA hybride :
    1. gagne si possible
    2. bloque si nécessaire
    3. sinon minimax + évaluation ML
    """
    moves = available_moves(board)
    if not moves:
        return -1

    human_player = "O" if ai_player == "X" else "X"

    # 1) gagner immédiatement
    win_move = find_immediate_winning_move(board, ai_player)
    if win_move is not None:
        return win_move

    # 2) bloquer immédiatement
    block_move = find_immediate_winning_move(board, human_player)
    if block_move is not None:
        return block_move

    best_move = moves[0]

    if ai_player == "X":
        best_score = -math.inf
        for move in moves:
            child = make_move(board, move, "X")
            score = minimax_hybrid(child, depth - 1, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = math.inf
        for move in moves:
            child = make_move(board, move, "O")
            score = minimax_hybrid(child, depth - 1, -math.inf, math.inf, True)
            if score < best_score:
                best_score = score
                best_move = move

    return best_move


# =========================================================
# MODAL FIN DE PARTIE
# =========================================================
class ResultModal(ctk.CTkToplevel):
    def __init__(self, parent: "ModernTicTacToeApp", result: str) -> None:
        super().__init__(parent)

        self.parent = parent
        self.result = result

        self.title("Fin de partie")
        self.geometry("430x260")
        self.resizable(False, False)

        self.update_idletasks()
        width = 430
        height = 260
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color="#020617")

        container = ctk.CTkFrame(
            self,
            corner_radius=24,
            fg_color="#0f172a",
            border_width=1,
            border_color="#1e293b",
        )
        container.pack(fill="both", expand=True, padx=16, pady=16)

        title = "Match nul" if result == "draw" else f"Victoire de {result}"
        subtitle = (
            "La partie se termine sans vainqueur."
            if result == "draw"
            else f"Le joueur {result} remporte cette manche."
        )

        ctk.CTkLabel(
            container,
            text="Fin de partie",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#93c5fd",
        ).pack(pady=(24, 8))

        ctk.CTkLabel(
            container,
            text=title,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#f8fafc",
        ).pack()

        ctk.CTkLabel(
            container,
            text=subtitle,
            font=ctk.CTkFont(size=14),
            text_color="#cbd5e1",
        ).pack(pady=(10, 24))

        actions = ctk.CTkFrame(container, fg_color="transparent")
        actions.pack()

        ctk.CTkButton(
            actions,
            text="Nouvelle partie",
            width=155,
            height=42,
            corner_radius=14,
            command=self.restart_game,
        ).grid(row=0, column=0, padx=8)

        ctk.CTkButton(
            actions,
            text="Quitter",
            width=155,
            height=42,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=self.quit_to_menu,
        ).grid(row=0, column=1, padx=8)

    def restart_game(self) -> None:
        self.destroy()
        self.parent.reset_game()

    def quit_to_menu(self) -> None:
        self.destroy()
        self.parent.show_mode_screen()


# =========================================================
# APPLICATION PRINCIPALE
# =========================================================
class ModernTicTacToeApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Morpion IA - NSDuo")
        self.geometry("980x700")
        self.minsize(980, 700)
        self.configure(fg_color="#020617")

        self.board: List[str] = [" "] * 9
        self.current_player = "X"
        self.mode = "human"
        self.game_over = False

        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=24, pady=24)

        self.show_mode_screen()

    # -----------------------------------------------------
    # OUTILS UI
    # -----------------------------------------------------
    def clear_container(self) -> None:
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def create_card(self, parent, fg: str = "#0b1220") -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent,
            corner_radius=26,
            fg_color=fg,
            border_width=1,
            border_color="#1e293b",
        )

    # -----------------------------------------------------
    # ECRAN CHOIX MODE
    # -----------------------------------------------------
    def show_mode_screen(self) -> None:
        self.clear_container()

        wrapper = ctk.CTkFrame(self.main_container, fg_color="transparent")
        wrapper.pack(fill="both", expand=True)

        ctk.CTkLabel(
            wrapper,
            text="MORPION IA",
            font=ctk.CTkFont(size=42, weight="bold"),
            text_color="#f8fafc",
        ).pack(pady=(40, 8))

        ctk.CTkLabel(
            wrapper,
            text="Choisissez votre mode de jeu",
            font=ctk.CTkFont(size=18),
            text_color="#94a3b8",
        ).pack(pady=(0, 32))

        grid = ctk.CTkFrame(wrapper, fg_color="transparent")
        grid.pack(expand=True)

        cards = [
            ("vs Human", "Deux joueurs s'affrontent sur la même interface.", "human"),
            ("vs IA (ML)", "L'IA choisit ses coups avec les modèles ML.", "ml"),
            ("vs IA (Hybride)", "Minimax profondeur 3 + évaluation ML.", "hybrid"),
        ]

        for i, (title, desc, mode) in enumerate(cards):
            card = self.create_card(grid, fg="#0b1220")
            card.grid(row=0, column=i, padx=14, pady=12, sticky="nsew")

            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color="#f8fafc",
            ).pack(padx=24, pady=(28, 12))

            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=14),
                text_color="#cbd5e1",
                wraplength=220,
                justify="center",
            ).pack(padx=20, pady=(0, 24))

            ctk.CTkButton(
                card,
                text="Jouer",
                width=180,
                height=42,
                corner_radius=14,
                command=lambda m=mode: self.start_game(m),
            ).pack(pady=(0, 28))

    # -----------------------------------------------------
    # ECRAN JEU
    # -----------------------------------------------------
    def start_game(self, mode: str) -> None:
        self.mode = mode
        self.board = [" "] * 9
        self.current_player = "X"
        self.game_over = False
        self.show_game_screen()

    def show_game_screen(self) -> None:
        self.clear_container()

        layout = ctk.CTkFrame(self.main_container, fg_color="transparent")
        layout.pack(fill="both", expand=True)

        left = ctk.CTkFrame(layout, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))

        right = ctk.CTkFrame(layout, fg_color="transparent", width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Carte principale du plateau
        board_card = self.create_card(left, fg="#0b1220")
        board_card.pack(fill="both", expand=True)

        topbar = ctk.CTkFrame(board_card, fg_color="transparent")
        topbar.pack(fill="x", padx=24, pady=(20, 8))

        ctk.CTkLabel(
            topbar,
            text=self.get_mode_label(),
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f8fafc",
        ).pack(side="left")

        ctk.CTkButton(
            topbar,
            text="Accueil",
            width=110,
            height=36,
            corner_radius=12,
            fg_color="#1e293b",
            hover_color="#334155",
            command=self.show_mode_screen,
        ).pack(side="right")

        self.status_label = ctk.CTkLabel(
            board_card,
            text=self.get_status_text(),
            font=ctk.CTkFont(size=16),
            text_color="#93c5fd",
        )
        self.status_label.pack(pady=(0, 14))

        grid_frame = ctk.CTkFrame(board_card, fg_color="transparent")
        grid_frame.pack(pady=(6, 28))

        self.cell_buttons: List[ctk.CTkButton] = []
        for i in range(9):
            btn = ctk.CTkButton(
                grid_frame,
                text="",
                width=130,
                height=130,
                corner_radius=22,
                font=ctk.CTkFont(size=42, weight="bold"),
                fg_color="#111827",
                hover_color="#1f2937",
                border_width=1,
                border_color="#1e293b",
                command=lambda idx=i: self.on_cell_click(idx),
            )
            btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)
            self.cell_buttons.append(btn)

        # sidebar actions
        actions_card = self.create_card(right, fg="#0b1220")
        actions_card.pack(fill="x", pady=(0, 14))

        ctk.CTkLabel(
            actions_card,
            text="Actions",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(anchor="w", padx=18, pady=(18, 12))

        ctk.CTkButton(
            actions_card,
            text="Nouvelle partie",
            height=42,
            corner_radius=14,
            command=self.reset_game,
        ).pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkButton(
            actions_card,
            text="Changer de mode",
            height=42,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=self.show_mode_screen,
        ).pack(fill="x", padx=18, pady=(0, 18))

        # sidebar infos
        info_card = self.create_card(right, fg="#0b1220")
        info_card.pack(fill="x")

        ctk.CTkLabel(
            info_card,
            text="Règles rapides",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f8fafc",
        ).pack(anchor="w", padx=18, pady=(18, 10))

        infos = [
            "• Le joueur X commence toujours",
            "• Alignez 3 symboles pour gagner",
            "• Le mode hybride anticipe mieux les pièges",
        ]
        for txt in infos:
            ctk.CTkLabel(
                info_card,
                text=txt,
                font=ctk.CTkFont(size=14),
                text_color="#cbd5e1",
                anchor="w",
                justify="left",
                wraplength=250,
            ).pack(anchor="w", padx=18, pady=4, fill="x")

        ctk.CTkLabel(info_card, text="", height=8).pack()

        self.update_board_ui()

    def get_mode_label(self) -> str:
        if self.mode == "human":
            return "Joueur vs Joueur"
        if self.mode == "ml":
            return "Joueur vs IA (ML)"
        return "Joueur vs IA (Hybride)"

    def get_status_text(self) -> str:
        if self.game_over:
            winner = check_winner(self.board)
            if winner:
                return f"Le joueur {winner} a gagné"
            return "Match nul"

        if self.mode == "human":
            return f"Tour de {self.current_player}"

        if self.current_player == "X":
            return "À vous de jouer"
        return "L'IA réfléchit..."

    def reset_game(self) -> None:
        self.board = [" "] * 9
        self.current_player = "X"
        self.game_over = False
        self.show_game_screen()

    # -----------------------------------------------------
    # LOGIQUE DE JEU
    # -----------------------------------------------------
    def on_cell_click(self, index: int) -> None:
        if self.game_over:
            return
        if self.board[index] != " ":
            return

        if self.mode == "human":
            self.play_human(index)
        else:
            self.play_vs_ai(index)

    def play_human(self, index: int) -> None:
        self.board[index] = self.current_player
        self.update_board_ui()

        if self.check_end_game():
            return

        self.current_player = "O" if self.current_player == "X" else "X"
        self.refresh_status()

    def play_vs_ai(self, index: int) -> None:
        if self.current_player != "X":
            return

        self.board[index] = "X"
        self.update_board_ui()

        if self.check_end_game():
            return

        self.current_player = "O"
        self.refresh_status()

        self.after(350, self.ai_turn)

    def ai_turn(self) -> None:
        if self.game_over:
            return

        moves = available_moves(self.board)
        if not moves:
            return

        if self.mode == "ml":
            move = best_move_ml(self.board, ai_player="O")
        else:
            move = best_move_hybrid(self.board, ai_player="O", depth=3)

        if move == -1:
            return

        self.board[move] = "O"
        self.update_board_ui()

        if self.check_end_game():
            return

        self.current_player = "X"
        self.refresh_status()

    def check_end_game(self) -> bool:
        winner = check_winner(self.board)
        if winner:
            self.game_over = True
            self.refresh_status()
            self.disable_board()
            self.after(200, lambda: ResultModal(self, winner))
            return True

        if is_full(self.board):
            self.game_over = True
            self.refresh_status()
            self.disable_board()
            self.after(200, lambda: ResultModal(self, "draw"))
            return True

        return False

    # -----------------------------------------------------
    # MISE A JOUR UI
    # -----------------------------------------------------
    def update_board_ui(self) -> None:
        if not hasattr(self, "cell_buttons"):
            return

        for i, btn in enumerate(self.cell_buttons):
            value = self.board[i]

            if value == "X":
                btn.configure(
                    text="X",
                    text_color="#60a5fa",
                    state="disabled",
                    fg_color="#111827",
                )
            elif value == "O":
                btn.configure(
                    text="O",
                    text_color="#c084fc",
                    state="disabled",
                    fg_color="#111827",
                )
            else:
                state = "normal" if not self.game_over else "disabled"
                btn.configure(
                    text="",
                    text_color="#ffffff",
                    state=state,
                    fg_color="#111827",
                )

        self.refresh_status()

    def disable_board(self) -> None:
        for btn in self.cell_buttons:
            btn.configure(state="disabled")

    def refresh_status(self) -> None:
        if hasattr(self, "status_label"):
            self.status_label.configure(text=self.get_status_text())
        if hasattr(self, "side_status_label"):
            self.side_status_label.configure(text=self.get_status_text())


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app = ModernTicTacToeApp()
    app.mainloop()