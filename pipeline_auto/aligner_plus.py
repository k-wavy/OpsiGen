import os
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBIO
from Bio import pairwise2
from Bio import SeqIO
from config import parse_args
from Bio.pairwise2 import format_alignment
import subprocess
from sequences import Sequences
import json

TEMP_CONFIG = "configs/mafft_test.json"

class Aligner:

    def __init__(self, config):
        self.seq, self.seq_name = Aligner._parse_fasta(config["fasta_path"])
        self.seq_name = self.seq_name.strip()
        self.db = Sequences(config)
        self.best_match, self.best_index = None, None
        self.aligning_index = config["aligning_index"]
        self.alignment_path = config["sequences"]
        self.seq_path = config["fasta_path"]
        self.mafft_id = config["mafft_id"]


    @staticmethod
    def _parse_fasta(fasta_file):
        with open(fasta_file, "r") as f:
            lines = f.readlines()

        return lines[1], lines[0][1:]

    def _align_to_entry(self, entry_number):
        curr_seq = self.db._get_relevant_line(entry_number)
        aligned = pairwise2.align.globalxx(self.seq, curr_seq)[0]

        return aligned

    def _align_to_known_index(self, aligning_index):
        print("Aligning the given sequence to the given entry in the DB")
        aligned = self._align_to_entry(aligning_index)
        best_match = aligned.seqA
        best_score = aligned.score

        return best_match, aligning_index

    def _align_to_best_match(self):
        print("Aligning the given sequence to the DB and finding best match:")
        best_index = 0
        best_match = ''
        best_score = 0
        for i in tqdm(range(self.db.length)):
            aligned = self._align_to_entry(i)
            if aligned.score > best_score:
                best_index = i
                best_match = aligned.seqA
                best_score = aligned.score

        print("Best index is", best_index)

        return best_match, best_index

    def align(self):
        if self.aligning_index > 0:
            best_match, best_index = self._align_to_known_index(self.aligning_index)
        else:
            best_match, best_index = self._align_to_best_match()

        return best_match, best_index

    @staticmethod
    def get_sequence_by_name(fasta_file, sequence_name):
        """
        Extract sequence from a multi-FASTA file by sequence name.

        Parameters:
        - fasta_file: path to the multi-FASTA file
        - sequence_name: name of the sequence to extract

        Returns:
        - sequence: string containing the sequence
        """
        with open(fasta_file, "r") as f:
            lines = f.readlines()
        seq_start_ind = [i for i in range(len(lines)) if lines[i].strip() == ">" + sequence_name][0] + 1
        i = 0
        res = ""
        while seq_start_ind + i < len(lines) and lines[seq_start_ind + i][0] != ">":
            res += lines[seq_start_ind + i].strip()
            i += 1
        return res

    @staticmethod
    def calc_single_pos(pos, seq):
        return len(seq[:pos].replace("-", ""))

    def get_amino_acids_of_aligned_sequence(self):
        new_align_path = "./mafft_alignments/" + self.mafft_id + ".fasta"
        mafft_command = f"./run_mafft.sh" \
                        f" {self.seq_path} {self.alignment_path} {new_align_path}"
        subprocess.run(mafft_command, shell=True)
        seq = Aligner.get_sequence_by_name(new_align_path, self.seq_name)
        positions = [Aligner.calc_single_pos(pos, seq) + (seq[pos-1] == "-") for pos in Sequences.POSITIONS]
        # self.best_match, self.best_index = self.align()
        # positions = self.db.get_interesting_positions(self.best_index)

        return positions


def main():
    pass
    # with open(TEMP_CONFIG, "r") as f:
    #     data = f.read()
    # config = json.loads(data)
    # aligner = Aligner(config)
    # print(aligner.get_amino_acids_of_aligned_sequence())


if __name__ == "__main__":
    main()
