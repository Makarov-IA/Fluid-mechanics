#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "functions/g_functions.hpp"

struct LogInfo
{
    bool enabled = true;
    std::string level = "info";
    int print_every_step = 100;
};

struct Config
{
    int nx = 128;
    int ny = 128;

    double t_max = 1.0;
    int n_time_steps = 1000;
    double nu = 1e-3;

    // Order: (up, right, down, left)
    std::array<double, 4> u{0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> v{0.0, 0.0, 0.0, 0.0};

    int g_function_id = 0;
    int save_every_step = 100;
    std::string save_dir = "data/results";

    LogInfo log_info;
};

inline std::string Trim(const std::string& s)
{
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first])))
    {
        ++first;
    }

    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1])))
    {
        --last;
    }

    return s.substr(first, last - first);
}

inline std::string StripComment(const std::string& line)
{
    const std::size_t hash_pos = line.find('#');
    if (hash_pos == std::string::npos)
    {
        return line;
    }
    return line.substr(0, hash_pos);
}

inline bool ParseBool(const std::string& value)
{
    if (value == "true" || value == "1" || value == "on")
    {
        return true;
    }
    if (value == "false" || value == "0" || value == "off")
    {
        return false;
    }

    throw std::runtime_error("Invalid bool value: " + value);
}

inline std::array<double, 4> ParseArray4(const std::string& value, const std::string& key)
{
    std::string cleaned = Trim(value);
    if (!cleaned.empty() && cleaned.front() == '[' && cleaned.back() == ']')
    {
        cleaned = cleaned.substr(1, cleaned.size() - 2);
    }

    std::array<double, 4> out{};
    std::stringstream ss(cleaned);
    std::string token;
    int idx = 0;

    while (std::getline(ss, token, ','))
    {
        if (idx >= 4)
        {
            throw std::runtime_error("Field " + key + " must contain exactly 4 numbers");
        }

        out[static_cast<std::size_t>(idx)] = std::stod(Trim(token));
        ++idx;
    }

    if (idx != 4)
    {
        throw std::runtime_error("Field " + key + " must contain exactly 4 numbers");
    }

    return out;
}

inline Config LoadConfigFromFile(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    Config cfg;
    bool has_u = false;
    bool has_v = false;

    std::string raw_line;
    int line_no = 0;

    while (std::getline(in, raw_line))
    {
        ++line_no;
        const std::string line = Trim(StripComment(raw_line));

        if (line.empty())
        {
            continue;
        }

        const std::size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos)
        {
            throw std::runtime_error("Invalid config line " + std::to_string(line_no) + ": expected key=value");
        }

        const std::string key = Trim(line.substr(0, eq_pos));
        const std::string value = Trim(line.substr(eq_pos + 1));

        try
        {
            if (key == "nx")
            {
                cfg.nx = std::stoi(value);
            }
            else if (key == "ny")
            {
                cfg.ny = std::stoi(value);
            }
            else if (key == "t_max")
            {
                cfg.t_max = std::stod(value);
            }
            else if (key == "n_time_steps")
            {
                cfg.n_time_steps = std::stoi(value);
            }
            else if (key == "nu")
            {
                cfg.nu = std::stod(value);
            }
            else if (key == "u")
            {
                cfg.u = ParseArray4(value, "u");
                has_u = true;
            }
            else if (key == "v")
            {
                cfg.v = ParseArray4(value, "v");
                has_v = true;
            }
            else if (key == "g_function_id")
            {
                cfg.g_function_id = std::stoi(value);
            }
            else if (key == "save_every_step")
            {
                cfg.save_every_step = std::stoi(value);
            }
            else if (key == "save_dir")
            {
                cfg.save_dir = value;
            }
            else if (key == "log_info.enabled")
            {
                cfg.log_info.enabled = ParseBool(value);
            }
            else if (key == "log_info.level")
            {
                cfg.log_info.level = value;
            }
            else if (key == "log_info.print_every_step")
            {
                cfg.log_info.print_every_step = std::stoi(value);
            }
            else
            {
                throw std::runtime_error("Unknown key: " + key);
            }
        }
        catch (const std::exception& ex)
        {
            throw std::runtime_error("Config parse error at line " + std::to_string(line_no) + ": " + ex.what());
        }
    }

    if (!has_u)
    {
        throw std::runtime_error("Missing required key: u");
    }
    if (!has_v)
    {
        throw std::runtime_error("Missing required key: v");
    }

    if (cfg.nx <= 1)
    {
        throw std::runtime_error("nx must be > 1");
    }
    if (cfg.ny <= 1)
    {
        throw std::runtime_error("ny must be > 1");
    }
    if (cfg.t_max <= 0.0)
    {
        throw std::runtime_error("t_max must be > 0");
    }
    if (cfg.n_time_steps <= 0)
    {
        throw std::runtime_error("n_time_steps must be > 0");
    }
    if (cfg.nu <= 0.0)
    {
        throw std::runtime_error("nu must be > 0");
    }
    if (cfg.save_every_step <= 0)
    {
        throw std::runtime_error("save_every_step must be > 0");
    }
    if (cfg.save_dir.empty())
    {
        throw std::runtime_error("save_dir must not be empty");
    }
    if (cfg.log_info.print_every_step <= 0)
    {
        throw std::runtime_error("log_info.print_every_step must be > 0");
    }

    (void)GetGFunctionById(cfg.g_function_id);

    return cfg;
}

inline GFunction BuildGFunction(const Config& cfg)
{
    return GetGFunctionById(cfg.g_function_id);
}

#endif // CONFIG_HPP
