#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::vector<std::string>> rows(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  std::vector<std::vector<std::string>> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<std::string> row;
    std::string token;
    while (fields >> token) row.push_back(token);
    result.push_back(row);
  }
  return result;
}

int main()
{
  std::ifstream metadata("geometry_only_geometry.json");
  assert(metadata.good());
  const std::string json((std::istreambuf_iterator<char>(metadata)), std::istreambuf_iterator<char>());
  assert(json.find("\"axis\": \"z\"") != std::string::npos);
  assert(json.find("\"pbc\": [1, 1, 1]") != std::string::npos);
  const auto bins = rows("geometry_only_bins.dat");
  const auto atoms = rows("geometry_only_groups.dat");
  assert(bins.size() == 4 && atoms.size() == 8);
  for (const auto& bin : bins) assert(bin.size() == 8);
  for (const auto& atom : atoms) assert(atom.size() == 11);
}
