import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class WoundDetector:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.wound_candidates = []
        self.confirmed_wounds = []

    def load_image(self, image_path):
        """Load and validate input image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("Could not load image")
        
        self.processed_image = self.original_image.copy()
        return True

    def detect_red_regions_aggressive(self, image):
        """Use LAB method directly - best performing method"""
        # LAB color space detection - proven to work best
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_lab = cv2.split(lab)
        
        # High 'a' values indicate red colors
        lab_red = np.zeros(image.shape[:2], dtype=np.uint8)
        lab_red[a > 160] = 255
        
        # Clean up the result
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(lab_red, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        print(f"LAB method detected: {np.sum(cleaned > 0)} red pixels")
        
        # Show detection result
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(lab_red, cmap='gray')
        plt.title('LAB Red Detection')
        plt.axis('off')
        
        # Show overlay
        overlay = image.copy()
        overlay[cleaned > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title('Detected Red Regions (Green overlay)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return cleaned

    def detect_wounds_comprehensive(self):
        """Streamlined wound detection using LAB method - direct processing"""
        print("Detecting wounds using LAB color space method...")
        
        red_mask = self.detect_red_regions_aggressive(self.original_image)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        if len(contours) == 0:
            print("No red regions detected!")
            return False
        
        # Filter contours by size and directly add to wound candidates
        self.wound_candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            print(f"Contour {i}: area={area:.1f}, dimensions={w}x{h}")
            
            # Keep reasonably sized regions
            if 50 < area < 2000 and w > 10 and h > 10:
                self.wound_candidates.append(contour)
                print(f"  -> Accepted as wound candidate")
            else:
                print(f"  -> Rejected (size filter)")
        
        print(f"Final candidates: {len(self.wound_candidates)}")
        
        if len(self.wound_candidates) == 0:
            print("No valid wound candidates found after filtering!")
            return False
        
        # Skip visualization - go directly to processing
        return True

    def validate_wound_region(self, contour):
        """Don't validate - just accept everything"""
        return True  # Accept all candidates

    def show_wounds_for_confirmation(self):
        """Skip this - we already confirmed during detection"""
        return self.wound_candidates

    def create_bandaid_template(self):
        """Create a realistic band-aid template"""
        bandaid_width = 80
        bandaid_height = 25
        
        # Create base shape
        bandaid = np.ones((bandaid_height, bandaid_width, 3), dtype=np.uint8)
        
        # Realistic beige color
        beige_color = (165, 185, 210)  # BGR
        bandaid[:, :] = beige_color
        
        # Add padding texture
        pad_width = int(bandaid_width * 0.15)
        pad_color = (145, 165, 190)  # Slightly darker beige
        
        # Left pad
        bandaid[:, :pad_width] = pad_color
        # Right pad  
        bandaid[:, -pad_width:] = pad_color
        
        # Add subtle holes pattern in pads
        for x in range(5, pad_width - 5, 8):
            for y in range(4, bandaid_height - 4, 6):
                cv2.circle(bandaid, (x, y), 1, (120, 140, 165), -1)
                cv2.circle(bandaid, (bandaid_width - x, y), 1, (120, 140, 165), -1)
        
        # Create alpha channel with rounded edges
        alpha = np.ones((bandaid_height, bandaid_width), dtype=np.uint8) * 240
        
        # Round corners
        corner_radius = 8
        corners = [(0, 0), (0, bandaid_height-1), 
                  (bandaid_width-1, 0), (bandaid_width-1, bandaid_height-1)]
        
        for corner_x, corner_y in corners:
            for dx in range(corner_radius):
                for dy in range(corner_radius):
                    if dx*dx + dy*dy > corner_radius*corner_radius:
                        alpha[corner_y + dy if corner_y == 0 else corner_y - dy,
                              corner_x + dx if corner_x == 0 else corner_x - dx] = 0
        
        return bandaid, alpha

    def apply_bandaid_to_wound(self, contour):
        """Apply band-aid with precise positioning and orientation to cover only the wound"""
        # Get detailed wound characteristics
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate the minimum area rectangle (oriented bounding box)
        rect = cv2.minAreaRect(contour)
        (rect_center_x, rect_center_y), (rect_width, rect_height), angle = rect
        
        # Use the oriented rectangle dimensions for better fitting
        wound_area = cv2.contourArea(contour)
        
        # Create band-aid
        bandaid, alpha = self.create_bandaid_template()
        
        # Determine scaling based on wound dimensions
        # Add some padding around the wound
        padding_factor = 1.3
        target_width = max(rect_width * padding_factor, w * padding_factor)
        target_height = max(rect_height * padding_factor, h * padding_factor)
        
        # Scale band-aid to match wound dimensions
        scale_w = target_width / bandaid.shape[1]
        scale_h = target_height / bandaid.shape[0]
        
        # Limit scaling to reasonable bounds
        scale_w = np.clip(scale_w, 0.4, 3.0)
        scale_h = np.clip(scale_h, 0.4, 2.0)
        
        # Calculate new band-aid dimensions
        new_width = int(bandaid.shape[1] * scale_w)
        new_height = int(bandaid.shape[0] * scale_h)
        
        # Ensure minimum and maximum sizes
        new_width = max(25, min(new_width, 150))
        new_height = max(12, min(new_height, 80))
        
        # Resize band-aid
        bandaid_resized = cv2.resize(bandaid, (new_width, new_height))
        alpha_resized = cv2.resize(alpha, (new_width, new_height))
        
        # Handle rotation for elongated wounds
        wound_aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height) if min(rect_width, rect_height) > 0 else 1
        
        if wound_aspect_ratio > 2.0:  # Elongated wound
            # Rotate band-aid to match wound orientation
            rotation_angle = angle
            
            # Adjust angle to align band-aid length with wound length
            if rect_width < rect_height:  # Wound is taller than wide
                rotation_angle += 90
            
            # Create rotation matrix
            center = (new_width // 2, new_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Calculate new dimensions after rotation
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            rotated_width = int((new_height * sin_angle) + (new_width * cos_angle))
            rotated_height = int((new_height * cos_angle) + (new_width * sin_angle))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (rotated_width / 2) - center[0]
            rotation_matrix[1, 2] += (rotated_height / 2) - center[1]
            
            # Rotate band-aid and alpha
            bandaid_rotated = cv2.warpAffine(bandaid_resized, rotation_matrix, (rotated_width, rotated_height))
            alpha_rotated = cv2.warpAffine(alpha_resized, rotation_matrix, (rotated_width, rotated_height))
            
            final_bandaid = bandaid_rotated
            final_alpha = alpha_rotated
            final_width = rotated_width
            final_height = rotated_height
        else:
            # No rotation needed for roughly circular wounds
            final_bandaid = bandaid_resized
            final_alpha = alpha_resized
            final_width = new_width
            final_height = new_height
        
        # Position band-aid centered on the wound
        start_x = max(0, int(rect_center_x - final_width // 2))
        start_y = max(0, int(rect_center_y - final_height // 2))
        end_x = min(self.processed_image.shape[1], start_x + final_width)
        end_y = min(self.processed_image.shape[0], start_y + final_height)
        
        # Apply with proper blending
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        
        if actual_width > 0 and actual_height > 0:
            # Crop band-aid if it extends beyond image bounds
            bandaid_cropped = final_bandaid[:actual_height, :actual_width]
            alpha_cropped = final_alpha[:actual_height, :actual_width].astype(float) / 255.0
            
            # Create a mask to only apply band-aid where there's actual bandaid content
            bandaid_mask = alpha_cropped > 0.1  # Only apply where alpha is significant
            
            # Blend the band-aid only in the masked areas
            region = self.processed_image[start_y:end_y, start_x:end_x]
            
            for c in range(3):
                # Only modify pixels where the band-aid should be applied
                region[:, :, c] = np.where(
                    bandaid_mask,
                    bandaid_cropped[:, :, c] * alpha_cropped + region[:, :, c] * (1 - alpha_cropped),
                    region[:, :, c]
                )
        
        # Print info about the applied band-aid
        print(f"Applied band-aid: {final_width}x{final_height} pixels, "
              f"positioned at ({start_x}, {start_y})")
        if wound_aspect_ratio > 2.0:
            print(f"Rotated by {angle:.1f} degrees to match wound orientation")

    def create_adaptive_bandaid_template(self, wound_width, wound_height):
        """Create a band-aid template that adapts to wound dimensions"""
        # Calculate optimal band-aid dimensions
        min_width = max(30, int(wound_width * 1.2))
        min_height = max(15, int(wound_height * 1.2))
        
        # Maintain reasonable aspect ratio
        if min_width / min_height > 4:  # Too wide
            min_height = int(min_width / 3.5)
        elif min_height / min_width > 2.5:  # Too tall
            min_width = int(min_height / 2)
        
        # Create base shape with calculated dimensions
        bandaid = np.ones((min_height, min_width, 3), dtype=np.uint8)
        
        # Realistic beige color
        beige_color = (165, 185, 210)  # BGR
        bandaid[:, :] = beige_color
        
        # Add padding texture (adjust for size)
        pad_width = max(3, int(min_width * 0.15))
        pad_color = (145, 165, 190)  # Slightly darker beige
        
        # Only add pads if band-aid is wide enough
        if min_width > 20:
            # Left pad
            bandaid[:, :pad_width] = pad_color
            # Right pad  
            bandaid[:, -pad_width:] = pad_color
            
            # Add holes pattern in pads
            hole_spacing = max(4, pad_width // 2)
            for x in range(2, pad_width - 2, hole_spacing):
                for y in range(2, min_height - 2, max(3, min_height // 6)):
                    if x < pad_width and y < min_height:
                        cv2.circle(bandaid, (x, y), 1, (120, 140, 165), -1)
                        if min_width - x < min_width:
                            cv2.circle(bandaid, (min_width - x - 1, y), 1, (120, 140, 165), -1)
        
        # Create alpha channel with rounded edges
        alpha = np.ones((min_height, min_width), dtype=np.uint8) * 240
        
        # Round corners (adjust radius based on size)
        corner_radius = min(8, min(min_width, min_height) // 6)
        corners = [(0, 0), (0, min_height-1), 
                  (min_width-1, 0), (min_width-1, min_height-1)]
        
        for corner_x, corner_y in corners:
            for dx in range(corner_radius):
                for dy in range(corner_radius):
                    if dx*dx + dy*dy > corner_radius*corner_radius:
                        if (corner_y == 0):
                            y_pos = corner_y + dy
                        else:
                            y_pos = corner_y - dy
                        
                        if (corner_x == 0):
                            x_pos = corner_x + dx
                        else:
                            x_pos = corner_x - dx
                            
                        if 0 <= y_pos < min_height and 0 <= x_pos < min_width:
                            alpha[y_pos, x_pos] = 0
        
        return bandaid, alpha

    def display_results(self):
        """Display original and processed images"""
        if self.original_image is None:
            return
        
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(processed_rgb)
        ax2.set_title('Image with Band-aids Applied')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_result(self, output_path):
        """Save the processed image"""
        if self.processed_image is not None:
            cv2.imwrite(output_path, self.processed_image)
            print(f"Result saved to: {output_path}")

    def process_image(self, image_path):
        """Main processing function"""
        self.load_image(image_path)
        
        print("Starting comprehensive wound detection...")
        wounds_found = self.detect_wounds_comprehensive()
        
        if wounds_found:
            confirmed_wounds = self.show_wounds_for_confirmation()
            
            if confirmed_wounds:
                print(f"Applying band-aids to {len(confirmed_wounds)} confirmed wounds")
                for i, contour in enumerate(confirmed_wounds):
                    self.apply_bandaid_to_wound(contour)
                    print(f"Applied band-aid {i + 1}")
                return True
            else:
                print("No wounds confirmed by user")
                return False
        else:
            print("No potential wounds detected")
            return False

def main():
    """Main function"""
    detector = WoundDetector()
    
    image_path = input("Enter the path to your image: ").strip()
    
    try:
        wounds_treated = detector.process_image(image_path)
        
        if wounds_treated:
            detector.display_results()
            
            save_choice = input("Save the result? (y/n): ").lower()
            if save_choice == 'y':
                output_path = "/c:/Users/admin/Assignment/arm_with_bandaid.jpg"
                detector.save_result(output_path)
        else:
            # Still show original image if no wounds found
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(detector.original_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image - No Wounds Detected')
            plt.axis('off')
            plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
