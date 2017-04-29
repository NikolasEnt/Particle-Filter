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

#define NUMBER_OF_PARTICLES 500 // Can be decreased (even 12 particles can pass the test)
#define EPS 0.001  // Just a small number

using namespace std;

// The Particle Filter functions
// Initialize all particles to first position
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	static default_random_engine gen;
    gen.seed(123);
    num_particles = NUMBER_OF_PARTICLES; // init number of particles to use
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	particles.resize(num_particles); // Resize the `particles` vector to fit desired number of particles
	weights.resize(num_particles);
	double init_weight = 1.0/num_particles;
	for (int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = init_weight;
	}	
	is_initialized = true;
}

// Add measurements to each particle and add random Gaussian noise.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
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

// Update the weights of each particle using a mult-variate Gaussian distribution.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	const double sigma_xx = std_landmark[0]*std_landmark[0];
	const double sigma_yy = std_landmark[1]*std_landmark[1];
	const double k = 2 * M_PI * std_landmark[0] * std_landmark[1];
	double dx = 0.0;
	double dy = 0.0;
	double sum_w = 0.0; // Sum of weights for future weights normalization
	for (int i = 0; i < num_particles; i++){
		double weight_no_exp = 0.0;
		const double sin_theta = sin(particles[i].theta);
		const double cos_theta = cos(particles[i].theta);
		for (int j = 0; j < observations.size(); j++){
			// Observation measurement transformations
			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = particles[i].x + (observations[j].x * cos_theta) - (observations[j].y * sin_theta);
			observation.y = particles[i].y + (observations[j].x * sin_theta) + (observations[j].y * cos_theta);
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
				weight_no_exp += dx * dx / sigma_xx + dy * dy / sigma_yy;
			}
			else {
				weight_no_exp += 100; // approx = 0 after exp()
			}
		}
		particles[i].weight = exp(-0.5*weight_no_exp); // calculate exp() after main computation in order to optimize the code
		sum_w += particles[i].weight;
	}
	// Weights normalization to sum(weights)=1
	for (int i = 0; i < num_particles; i++){
		particles[i].weight /= sum_w * k;
		weights[i] = particles[i].weight;
	}	
}

// Resampling particles with replacement with probability proportional to their weight. 
void ParticleFilter::resample() {
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
