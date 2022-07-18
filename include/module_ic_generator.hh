// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#pragma once

#include <memory>

#include <config_file.hh>
#include <random_plugin.hh>
#include <output_plugin.hh>
#include <cosmology_calculator.hh>

namespace ic_generator{

    int run( config_file& the_config );
    
    int initialise( config_file& the_config );

    void reset();

    extern std::unique_ptr<RNG_plugin> the_random_number_generator;
    extern std::unique_ptr<output_plugin> the_output_plugin;
    extern std::unique_ptr<cosmology::calculator>  the_cosmo_calc;

}
