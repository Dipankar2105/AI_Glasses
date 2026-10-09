import os, sys, json, urllib.request
from PIL import Image
import numpy as np

DATASET_DIR = "data/external/notebooks"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")

NOTEBOOK_SAMPLES = [
    # --- DEVELOPMENT SPLIT (15 samples) ---
    {
        "id": "nb_01",
        "file_name": "nb_01_ds_stack.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_10.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Stack Operations & LIFO Structure",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Stack is a linear data structure that follows LIFO principle. The operations are Push and Pop. Top points to top element."
    },
    {
        "id": "nb_02",
        "file_name": "nb_02_ds_queue.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_11.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Queue Operations & FIFO Principle",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Queue follows First In First Out (FIFO) order. Enqueue inserts at rear and Dequeue removes from front."
    },
    {
        "id": "nb_03",
        "file_name": "nb_03_ds_linkedlist.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_12.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Linked List Node & Pointer Traversals",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Singly Linked List node contains data and next pointer. Head points to first node and tail points to NULL."
    },
    {
        "id": "nb_04",
        "file_name": "nb_04_ds_trees.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_13.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Binary Search Tree Properties",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Binary Search Tree property: left child < root < right child. Inorder traversal of BST gives sorted keys."
    },
    {
        "id": "nb_05",
        "file_name": "nb_05_algo_recurrence.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DAA_SHORT_NOTE/DAA_SHORT_NOTE_page_2.jpg",
        "category": "algorithms",
        "subject": "Algorithms",
        "topic": "Recurrence Relations & Time Complexity",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Divide and Conquer Recurrence T(n) = 2T(n/2) + O(n). By Master Theorem Case 2, T(n) = O(n log n)."
    },
    {
        "id": "nb_06",
        "file_name": "nb_06_algo_greedy.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DAA_SHORT_NOTE/DAA_SHORT_NOTE_page_3.jpg",
        "category": "algorithms",
        "subject": "Algorithms",
        "topic": "Greedy Knapsack & Optimal Substructure",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Fractional Knapsack problem is solved by Greedy method. Sort items by value to weight ratio in descending order."
    },
    {
        "id": "nb_07",
        "file_name": "nb_07_dbms_er.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DBMS/DBMS_page_10.jpg",
        "category": "database_systems",
        "subject": "Database Systems",
        "topic": "Entity-Relationship Model & Primary Keys",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Entity Relationship Diagram: Entity represented by rectangle, attribute by oval, and relationship by diamond."
    },
    {
        "id": "nb_08",
        "file_name": "nb_08_dbms_normalization.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DBMS/DBMS_page_11.jpg",
        "category": "database_systems",
        "subject": "Database Systems",
        "topic": "Functional Dependencies & 3NF Normalization",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Normalization reduces data redundancy. 1NF eliminates repeating groups, 2NF removes partial dependency."
    },
    {
        "id": "nb_09",
        "file_name": "nb_09_math_ode.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Differntial_Equations-1/Differntial_Equations-1_page_10.jpg",
        "category": "mathematics",
        "subject": "Calculus",
        "topic": "First Order Linear Differential Equations",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "First order linear differential equation dy/dx + P(x)y = Q(x). Integrating Factor IF = exp(integral P(x) dx)."
    },
    {
        "id": "nb_10",
        "file_name": "nb_10_math_exact.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Differntial_Equations-1/Differntial_Equations-1_page_11.jpg",
        "category": "mathematics",
        "subject": "Calculus",
        "topic": "Exact Equations & Integrating Factors",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Exact Differential Equation M dx + N dy = 0 where dM/dy = dN/dx. Solution is integral M dx + integral N(free of x) dy = C."
    },
    {
        "id": "nb_11",
        "file_name": "nb_11_physics_torque.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Ch_-7_Rotational_motion/Ch_-7_Rotational_motion_page_34.jpg",
        "category": "physics",
        "subject": "Physics",
        "topic": "Torque & Angular Momentum Equations",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Torque tau = r x F = I alpha. Angular momentum L = I omega. Conservation of angular momentum holds when external torque is zero."
    },
    {
        "id": "nb_12",
        "file_name": "nb_12_physics_inertia.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Ch_-7_Rotational_motion/Ch_-7_Rotational_motion_page_35.jpg",
        "category": "physics",
        "subject": "Physics",
        "topic": "Parallel Axis Theorem & Kinetic Energy",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Moment of Inertia I = sum m_i r_i^2. Parallel axis theorem I = I_cm + M d^2. Rotational Kinetic Energy K = 1/2 I omega^2."
    },
    {
        "id": "nb_13",
        "file_name": "nb_13_chem_ph.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent_page_10.jpg",
        "category": "chemistry",
        "subject": "Chemistry",
        "topic": "Acid-Base Equilibrium & pH Calculations",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Arrhenius acid produces H+ ions in aqueous solution. pH is defined as negative logarithm of hydrogen ion concentration: pH = -log[H+]."
    },
    {
        "id": "nb_14",
        "file_name": "nb_14_chem_buffers.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent_page_11.jpg",
        "category": "chemistry",
        "subject": "Chemistry",
        "topic": "Henderson-Hasselbalch Equation & Buffer Action",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Henderson-Hasselbalch equation: pH = pKa + log([Conjugate Base] / [Weak Acid]). Buffer resists changes in pH."
    },
    {
        "id": "nb_15",
        "file_name": "nb_15_digital_logic.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Digital_systems/Digital_systems_page_10.jpg",
        "category": "electronics",
        "subject": "Digital Electronics",
        "topic": "Boolean Algebra & Logic Gate Truth Tables",
        "split": "Dev",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "De Morgan's Laws: (A + B)' = A' . B' and (A . B)' = A' + B'. NAND and NOR are universal gates."
    },

    # --- HELD-OUT TEST SPLIT (15 samples) ---
    {
        "id": "nb_16",
        "file_name": "nb_16_ds_graph.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_14.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Graph Representation (Matrix vs Adjacency List)",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Graph representation: Adjacency Matrix requires O(V^2) space. Adjacency List requires O(V + E) space."
    },
    {
        "id": "nb_17",
        "file_name": "nb_17_ds_hashing.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DATA_STRUCTURE/DATA_STRUCTURE_page_15.jpg",
        "category": "data_structures",
        "subject": "Computer Science",
        "topic": "Hash Functions & Collision Resolution",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Hash collisions are resolved by Chaining or Open Addressing (Linear Probing, Quadratic Probing, Double Hashing)."
    },
    {
        "id": "nb_18",
        "file_name": "nb_18_algo_dynamic.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DAA_SHORT_NOTE/DAA_SHORT_NOTE_page_4.jpg",
        "category": "algorithms",
        "subject": "Algorithms",
        "topic": "Longest Common Subsequence (LCS)",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Dynamic Programming characteristics: Overlapping Subproblems and Optimal Substructure. LCS table computed in O(m*n) time."
    },
    {
        "id": "nb_19",
        "file_name": "nb_19_algo_mst.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DAA_SHORT_NOTE/DAA_SHORT_NOTE_page_5.jpg",
        "category": "algorithms",
        "subject": "Algorithms",
        "topic": "Kruskal's & Prim's Minimum Spanning Tree",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Minimum Spanning Tree connects all vertices with minimum edge weight. Kruskal uses Disjoint Set Union O(E log V)."
    },
    {
        "id": "nb_20",
        "file_name": "nb_20_dbms_transactions.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DBMS/DBMS_page_12.jpg",
        "category": "database_systems",
        "subject": "Database Systems",
        "topic": "ACID Properties & Two-Phase Locking",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "ACID properties: Atomicity, Consistency, Isolation, Durability. 2PL ensures conflict serializability."
    },
    {
        "id": "nb_21",
        "file_name": "nb_21_dbms_indexing.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/DBMS/DBMS_page_13.jpg",
        "category": "database_systems",
        "subject": "Database Systems",
        "topic": "B+ Tree Indexing & File Organization",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "B+ Tree index stores all keys at leaf level linked in sequential order. Search complexity is O(log_B N)."
    },
    {
        "id": "nb_22",
        "file_name": "nb_22_math_laplace.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Differntial_Equations-1/Differntial_Equations-1_page_12.jpg",
        "category": "mathematics",
        "subject": "Calculus",
        "topic": "Laplace Transforms & Shifting Theorems",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Laplace Transform L{f(t)} = integral from 0 to inf exp(-st) f(t) dt. First Shifting Theorem L{exp(at) f(t)} = F(s-a)."
    },
    {
        "id": "nb_23",
        "file_name": "nb_23_math_series.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Differntial_Equations-1/Differntial_Equations-1_page_13.jpg",
        "category": "mathematics",
        "subject": "Calculus",
        "topic": "Power Series Solutions of ODEs",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Frobenius method for differential equation with regular singular point: assume solution y = sum a_n x^(n+r)."
    },
    {
        "id": "nb_24",
        "file_name": "nb_24_physics_rolling.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Ch_-7_Rotational_motion/Ch_-7_Rotational_motion_page_36.jpg",
        "category": "physics",
        "subject": "Physics",
        "topic": "Pure Rolling on Incline without Slipping",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Pure rolling condition: v_cm = R omega. Total kinetic energy in rolling = 1/2 M v_cm^2 (1 + k^2/R^2)."
    },
    {
        "id": "nb_25",
        "file_name": "nb_25_physics_precession.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Ch_-7_Rotational_motion/Ch_-7_Rotational_motion_page_37.jpg",
        "category": "physics",
        "subject": "Physics",
        "topic": "Angular Momentum Vector & Gyroscopic Precession",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Precession angular velocity Omega_p = tau / L = M g r / (I omega). Direction of angular momentum vector follows right hand thumb rule."
    },
    {
        "id": "nb_26",
        "file_name": "nb_26_chem_titration.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent_page_12.jpg",
        "category": "chemistry",
        "subject": "Chemistry",
        "topic": "Strong Acid Strong Base Titration Curve",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Equivalence point in strong acid strong base titration is at pH 7.0. Phenolphthalein indicator changes color in pH range 8.2 to 10.0."
    },
    {
        "id": "nb_27",
        "file_name": "nb_27_chem_hydrolysis.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent/Acid_Base_Chemistry_Handwritten_Notes_-_ChemContent_page_13.jpg",
        "category": "chemistry",
        "subject": "Chemistry",
        "topic": "Salt Hydrolysis Constant & Degree of Hydrolysis",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Salt of weak acid and strong base undergoes anionic hydrolysis: pH = 7 + 1/2 pKa + 1/2 log C. Solution is basic."
    },
    {
        "id": "nb_28",
        "file_name": "nb_28_digital_kmap.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Digital_systems/Digital_systems_page_11.jpg",
        "category": "electronics",
        "subject": "Digital Electronics",
        "topic": "4-Variable Karnaugh Map Simplification",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Karnaugh Map (K-map) minimizes Boolean expressions by grouping adjacent 1s in powers of 2 (pairs, quads, octets)."
    },
    {
        "id": "nb_29",
        "file_name": "nb_29_digital_flipflops.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Digital_systems/Digital_systems_page_12.jpg",
        "category": "electronics",
        "subject": "Digital Electronics",
        "topic": "JK & D Flip Flop State Transitions",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "JK Flip-Flop characteristic equation: Q(next) = J Q' + K' Q. Race around condition occurs when J=1, K=1 and pulse width is large."
    },
    {
        "id": "nb_30",
        "file_name": "nb_30_digital_counters.jpg",
        "url": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge/resolve/main/Task_1/Images/Digital_systems/Digital_systems_page_13.jpg",
        "category": "electronics",
        "subject": "Digital Electronics",
        "topic": "Synchronous vs Asynchronous Counters",
        "split": "HeldOut",
        "provenance": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
        "license": "CC BY 4.0",
        "transcription": "Mod-N counter has N distinct states. Synchronous counter triggers all flip-flops simultaneously with common clock."
    }
]

def acquire_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("Acquiring 30 genuine notebook page images from ICDAR 2025 Benchmark...")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    for idx, s in enumerate(NOTEBOOK_SAMPLES):
        dest = os.path.join(DATASET_DIR, s["file_name"])
        if not os.path.exists(dest):
            print(f"Downloading [{idx+1}/30] {s['id']}: {s['file_name']} ...")
            req = urllib.request.Request(s["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp, open(dest, 'wb') as out_f:
                out_f.write(resp.read())
                
        # Validate image integrity
        with Image.open(dest) as img:
            img.verify()
        with Image.open(dest) as img:
            s["dimensions"] = list(img.size)
            s["channels"] = len(img.getbands())
            
    print(f"\nWriting dataset manifest to {METADATA_PATH} ...")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        for s in NOTEBOOK_SAMPLES:
            f.write(json.dumps(s) + '\n')
            
    print(f"Successfully acquired and verified {len(NOTEBOOK_SAMPLES)} genuine notebook samples.")

if __name__ == '__main__':
    acquire_dataset()
