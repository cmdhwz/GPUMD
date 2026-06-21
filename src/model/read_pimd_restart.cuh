#pragma once

class Atom;
class Box;

void read_pimd_restart(const char* filename, int expected_number_of_beads, Box& box, Atom& atom);
