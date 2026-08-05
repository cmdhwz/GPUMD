#pragma once

class Atom;
class Box;

struct PIMD_Restart_Metadata
{
  bool has_temperature = false;
  double temperature = 0.0;
};

void read_pimd_restart(
  const char* filename,
  int expected_number_of_beads,
  Box& box,
  Atom& atom,
  PIMD_Restart_Metadata* metadata = nullptr);
