#include "FileMHD.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>


#if defined _WIN32
  const char kPathSeperator = '\\';
#else
  const char kPathSeperator = '/';
#endif


void FileMHD::Strip(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());  // rstrip
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));  // lstrip
}


bool FileMHD::Read(std::vector<int> &dimensions, std::vector<float> &values, std::vector<double> &spacing) {

  // load header from file
  std::map<std::string, std::string> header_in;
  std::string line;
  std::ifstream file_mhd(filename.c_str());
  if (!file_mhd.is_open()) {
    std::cerr << "FileMHD::Read: \"" << filename << "\" not found" << std::endl;
    return false;
  }
  while (file_mhd.good()) {
     std::getline(file_mhd, line);
     // skip empty and commented lines
     if (line.empty() || line[0] == '#') continue;
     std::string key = line.substr(0, line.find('='));
     std::string value = line.substr(line.find('=') + 1, line.length());
     Strip(key);
     Strip(value);
     // handle case variations of keys
     std::transform(key.begin(), key.end(), key.begin(), ::tolower);
     header_in[key] = value;
  }
  file_mhd.close();

  // typecast header tags to native types
  std::string tag;
  std::map<std::string, std::string>::const_iterator it;
  for (it = header_in.begin(); it != header_in.end(); ++it) {
    tag = it->first;
    if (tag == "ndims") {
      std::istringstream stream(it->second);
      stream >> header.NDims;
    } else if (tag == "compresseddata") {
      std::string value = it->second;
      std::transform(value.begin(), value.end(), value.begin(), ::tolower);
      header.CompressedData = value == "true";
      if (header.CompressedData) {
        std::cerr << "FileMHD::Read: compressed images are not supported" << std::endl;
        return false;
      }
    } else if (tag == "elementspacing") {
      std::istringstream stream(it->second);
      double value;
      while(stream >> value) header.ElementSpacing.push_back(value);
    } else if (tag == "dimsize") {
      std::istringstream stream(it->second);
      int value;
      while(stream >> value) header.DimSize.push_back(value);
    } else if (tag == "headersize") {
      std::istringstream stream(it->second);
      stream >> header.HeaderSize;
    } else if (tag == "elementtype") {
      std::string value = it->second;
      std::transform(value.begin(), value.end(), value.begin(), ::toupper);
      header.ElementType = value;
    } else if (tag == "elementdatafile") {
      header.ElementDataFile = it->second;
    }
  }

  // load image from file
  const unsigned int size = std::accumulate(header.DimSize.begin(), header.DimSize.end(), 1, std::multiplies<int>());
  std::string filename_raw = filename.substr(0, filename.rfind(kPathSeperator) + 1) + header.ElementDataFile;
  std::ifstream file_raw(filename_raw.c_str(), std::fstream::binary);
  if (!file_raw.is_open()) {
    std::cerr << "FileMHD::Read: \"" << filename_raw << "\" not found" << std::endl;
    return false;
  }
  file_raw.seekg(header.HeaderSize, std::ios::beg);
  if (header.ElementType == "MET_CHAR") {
    ReadCast<char,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_UCHAR") {
    ReadCast<unsigned char,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_SHORT") {
    ReadCast<short,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_USHORT") {
    ReadCast<unsigned short,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_INT") {
    ReadCast<int,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_UINT") {
    ReadCast<unsigned int,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_FLOAT") {
    ReadCast<float,float>(file_raw, values, size);
  } else if (header.ElementType == "MET_DOUBLE") {
    ReadCast<double,float>(file_raw, values, size);
  } else {
    std::cerr << "FileMHD::Read: ElementType \"" << header.ElementType << "\" is not supported" << std::endl;
    return false;
  }
  file_raw.close();

  dimensions = header.DimSize;
  spacing = header.ElementSpacing;
  return true;
}


bool FileMHD::Write(const std::vector<int> &dimensions, const std::vector<float> &values, const std::vector<double> &spacing) {
  // set required header tags?
  // set input header tags?
  // set ElementDataFile?
  const unsigned int size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>());

  // save image to file
  std::string filename_raw = filename.substr(0, filename.rfind('.')) + ".raw";
  std::string basename_raw = filename_raw.substr(filename_raw.rfind(kPathSeperator) + 1, filename_raw.length());
  std::ofstream file_raw(filename_raw.c_str(), std::fstream::binary);
  file_raw.write(reinterpret_cast<const char*>(&(values.front())), size * sizeof(float));
  file_raw.close();

  // typecast header tags to strings?

  // save header to file
  std::ofstream file_mhd(filename.c_str());
  file_mhd << "ObjectType = Image" << std::endl;
  file_mhd << "NDims = 3" << std::endl;
  file_mhd << "BinaryData = True" << std::endl;
  file_mhd << "BinaryDataByteOrderMSB = False" << std::endl;
  file_mhd << "ElementSpacing = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;
  file_mhd << "DimSize = " << dimensions[0] << " " << dimensions[1] << " " << dimensions[2] << std::endl;
  file_mhd << "ElementType = MET_FLOAT" << std::endl;
  file_mhd << "ElementDataFile = " << basename_raw << std::endl;
  file_mhd.close();

  // return header as a dictionary with unused tags removed?

  return true;
}
