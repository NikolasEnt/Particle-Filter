/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random> // Need this for sampling from distributions
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

#define NUMBER_OF_PARTICLES 500
#define EPS 0.0001  // Just a small number

using namespace std;

// Observation measurement transformations
LandmarkObs convert_coord(Particle p, LandmarkObs lm, const double sin_theta, const double cos_theta) {
    LandmarkObs new_lm;
    new_lm.id = lm.id;
    new_lm.x = p.x + (lm.x * cos_theta) - (lm.y * sin_theta);
    new_lm.y = p.y + (lm.x * sin_theta) + (lm.y * cos_theta);
    return new_lm;
}
// The Particle Filter functions
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	static default_random_engine gen;
    gen.seed(123);
    num_particles = NUMBER_OF_PARTICLES; // init number of particles to use
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	particles.resize(num_particles); // Resize the `particles` vector to fit desired number of particles
	weights.resize(num_particles);
	double init_weight = 1.0/num_particles; // To save computation
	for (int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = init_weight;
	}	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Some constants to save computation power
	const double vel_d_t = velocity * delta_t;
	const double yaw_d_t = yaw_rate * delta_t;
	const double vel_yaw = velocity/yaw_rate;
	static default_random_engine gen;
    gen.seed(321);
    normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	for (int i = 0; i < num_particles; i++){
        if (fabs(yaw_rate) < EPS){
            particles[i].x += vel_d_t * cos(particles[i].theta);
            particles[i].y += vel_d_t * sin(particles[i].theta);
            // particles[i].theta unchanged if yaw_rate is too small
        }
        else{
            const double theta_new = particles[i].theta + yaw_d_t;
            particles[i].x += vel_yaw * (sin(theta_new) - sin(particles[i].theta));
            particles[i].y += vel_yaw * (-cos(theta_new) + cos(particles[i].theta));
            particles[i].theta = theta_new;
        }
        // Add random Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	const double sigma_xx = std_landmark[0]*std_landmark[0];
	const double sigma_yy = std_landmark[1]*std_landmark[1];
	const double k = 2 * M_PI * std_landmark[0] * std_landmark[1];
	double dx = 0.0;
	double dy = 0.0;
	double sum_w = 0.0; // Sum of weights for future weights normalization
	for (int i = 0; i < num_particles; i++){
		double weight = 1.0;
		const double sin_theta = sin(particles[i].theta);
		const double cos_theta = cos(particles[i].theta);
		for (int j = 0; j < observations.size(); j++){
			LandmarkObs observation = convert_coord(particles[i], observations[j], sin_theta, cos_theta); // Convert coordinates
			// Unefficient way for observation asossiation to landmarks. It can be improved.
			bool in_range = false;
			Map::single_landmark_s nearest_lm;
            double nearest_dist = 10000000.0; // A big number
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                Map::single_landmark_s cond_lm = map_landmarks.landmark_list[k];
                double distance = dist(cond_lm.x_f, cond_lm.y_f, observation.x, observation.y);  // Calculate the Euclidean distance between two 2D points
                if (distance < nearest_dist) {
                    nearest_dist = distance;
                    nearest_lm = cond_lm;
                    if (distance < sensor_range){
						in_range = true;
					}
                }
            }
            if (in_range){
				dx = observation.x-nearest_lm.x_f;
				dy = observation.y-nearest_lm.y_f;
				weight *= exp(-0.5 * (dx * dx / sigma_xx + dy * dy / sigma_yy));
			}
			else {
				weight *= 0.0;
			}
            
		}
		particles[i].weight = weight;
		sum_w += weight;
	}
	
	// Weights normalization to sum(weights)=1
	for (int i = 0; i < num_particles; i++){
		particles[i].weight /= (sum_w * k);
		weights[i] = particles[i].weight;
	}
	
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    static default_random_engine gen;
    gen.seed(123);
    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        new_particles[i] = particles[dist_particles(gen)];
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
