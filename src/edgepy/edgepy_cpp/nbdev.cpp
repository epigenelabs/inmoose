#include "glm.h"

/* Function to calculate the deviance. Note the protection for very large mu*phi (where we
 * use a gamma instead) or very small mu*phi (where we use the Poisson instead). This
 * approximation protects against numerical instability introduced by subtracting
 * a very large log value in (log mu) with another very large logarithm (log mu+1/phi).
 * We need to consider the 'phi' as the approximation is only good when the product is
 * very big or very small.
 */

const double one_tenthousandth=std::pow(10, -4.0);
const double mildly_low_value=std::pow(10, -8.0);

double compute_unit_nb_deviance (double y, double mu, double phi) {
	// We add a small value to protect against zero during division and logging.
    y+=mildly_low_value;
    mu+=mildly_low_value;

    /* Calculating the deviance using either the Poisson (small phi*mu), the Gamma (large) or NB (everything else).
     * Some additional work is put in to make the transitions between families smooth.
     */
    if (phi < one_tenthousandth) {
		const double resid = y - mu;
		return 2 * ( y * std::log(y/mu) - resid - 0.5*resid*resid*phi*(1+phi*(2/3*resid-y)) );
    } else {
		const double product=mu*phi;
		if (product > one_million) {
            return 2 * ( (y - mu)/mu - std::log(y/mu) ) * mu/(1+product);
        } else {
			const double invphi=1/phi;
            return 2 * (y * std::log( y/mu ) + (y + invphi) * std::log( (mu + invphi)/(y + invphi) ) );
        }
	}
}

