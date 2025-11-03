import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

class EmailService:
    """
    Servicio para envío de correos electrónicos mediante SMTP.
    Soporta configuración mediante variables de entorno.
    """
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.from_name = os.getenv("FROM_NAME", "Sistema de Propiedades")
        self.enabled = os.getenv("EMAIL_ENABLED", "true").lower() == "true"
        
    def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None
    ) -> bool:
        """
        Enviar un correo electrónico.
        
        Args:
            to_email: Dirección de correo del destinatario
            subject: Asunto del correo
            body_text: Contenido en texto plano
            body_html: Contenido en HTML (opcional)
            
        Returns:
            bool: True si se envió exitosamente, False en caso contrario
        """
        if not self.enabled:
            print(f"📧 Email deshabilitado. No se enviará email a {to_email}")
            return False
            
        if not self.smtp_user or not self.smtp_password:
            print("⚠️ Credenciales SMTP no configuradas. No se puede enviar email.")
            return False
            
        try:
            # Crear mensaje
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # Adjuntar versión texto plano
            part_text = MIMEText(body_text, 'plain', 'utf-8')
            msg.attach(part_text)
            
            # Adjuntar versión HTML si existe
            if body_html:
                part_html = MIMEText(body_html, 'html', 'utf-8')
                msg.attach(part_html)
            
            # Conectar al servidor SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"✅ Email enviado exitosamente a {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Error al enviar email a {to_email}: {e}")
            return False
    
    def send_payment_confirmation(
        self,
        to_email: str,
        user_name: str,
        request_id: str,
        property_url: str,
        amount: float,
        authorization_code: Optional[str] = None
    ) -> bool:
        """
        Enviar correo de confirmación de pago.
        
        Args:
            to_email: Email del usuario
            user_name: Nombre del usuario
            request_id: ID de la solicitud
            property_url: URL de la propiedad
            amount: Monto pagado
            authorization_code: Código de autorización (opcional)
            
        Returns:
            bool: True si se envió exitosamente
        """
        subject = "✅ Confirmación de Pago - Reserva de Visita"
        
        # Versión texto plano
        body_text = f"""
Hola {user_name},

¡Tu pago ha sido procesado exitosamente!

Detalles de la transacción:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ID de Solicitud: {request_id}
• Propiedad: {property_url}
• Monto: ${amount:,.2f}
{f'• Código de Autorización: {authorization_code}' if authorization_code else ''}
• Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Tu reserva de visita ha sido confirmada y está en proceso de validación.
Recibirás una notificación adicional cuando la solicitud sea aceptada por el vendedor.

Puedes ver el estado de tu solicitud en tu historial de compras.

Gracias por usar nuestro servicio.

Saludos,
Sistema de Propiedades
        """
        
        # Versión HTML
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
        .content {{ background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
        .details {{ background-color: white; padding: 15px; margin: 15px 0; border-left: 4px solid #4CAF50; }}
        .detail-item {{ padding: 5px 0; }}
        .detail-label {{ font-weight: bold; color: #555; }}
        .footer {{ text-align: center; padding: 15px; color: #777; font-size: 12px; }}
        .success-icon {{ font-size: 48px; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="success-icon">✅</div>
            <h1 style="margin: 0;">Pago Confirmado</h1>
        </div>
        <div class="content">
            <p>Hola <strong>{user_name}</strong>,</p>
            <p>¡Tu pago ha sido procesado exitosamente!</p>
            
            <div class="details">
                <h3 style="margin-top: 0; color: #4CAF50;">Detalles de la Transacción</h3>
                <div class="detail-item">
                    <span class="detail-label">ID de Solicitud:</span> {request_id}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Propiedad:</span> {property_url}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Monto:</span> ${amount:,.2f}
                </div>
                {f'<div class="detail-item"><span class="detail-label">Código de Autorización:</span> {authorization_code}</div>' if authorization_code else ''}
                <div class="detail-item">
                    <span class="detail-label">Fecha:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            
            <p>Tu reserva de visita ha sido confirmada y está en proceso de validación.</p>
            <p>Recibirás una notificación adicional cuando la solicitud sea aceptada por el vendedor.</p>
            <p>Puedes ver el estado de tu solicitud en tu historial de compras.</p>
            
            <p style="margin-top: 20px;">Gracias por usar nuestro servicio.</p>
        </div>
        <div class="footer">
            <p>Este es un correo automático, por favor no respondas a este mensaje.</p>
            <p>&copy; {datetime.now().year} Sistema de Propiedades</p>
        </div>
    </div>
</body>
</html>
        """
        
        return self.send_email(to_email, subject, body_text.strip(), body_html)
    
    def send_payment_accepted(
        self,
        to_email: str,
        user_name: str,
        request_id: str,
        property_url: str,
        amount: float
    ) -> bool:
        """
        Enviar correo cuando el pago es aceptado por el vendedor.
        
        Args:
            to_email: Email del usuario
            user_name: Nombre del usuario
            request_id: ID de la solicitud
            property_url: URL de la propiedad
            amount: Monto pagado
            
        Returns:
            bool: True si se envió exitosamente
        """
        subject = "🎉 ¡Solicitud Aceptada! - Reserva de Visita"
        
        # Versión texto plano
        body_text = f"""
Hola {user_name},

¡Excelentes noticias! Tu solicitud de visita ha sido aceptada por el vendedor.

Detalles de la reserva:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ID de Solicitud: {request_id}
• Propiedad: {property_url}
• Monto: ${amount:,.2f}
• Estado: ACEPTADA
• Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

El monto de ${amount:,.2f} ha sido debitado de tu wallet.

Puedes descargar tu comprobante de pago desde tu historial de compras.

¡Disfruta tu visita!

Saludos,
Sistema de Propiedades
        """
        
        # Versión HTML
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #2196F3; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
        .content {{ background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
        .details {{ background-color: white; padding: 15px; margin: 15px 0; border-left: 4px solid #2196F3; }}
        .detail-item {{ padding: 5px 0; }}
        .detail-label {{ font-weight: bold; color: #555; }}
        .footer {{ text-align: center; padding: 15px; color: #777; font-size: 12px; }}
        .celebration-icon {{ font-size: 48px; margin-bottom: 10px; }}
        .status-badge {{ background-color: #4CAF50; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="celebration-icon">🎉</div>
            <h1 style="margin: 0;">¡Solicitud Aceptada!</h1>
        </div>
        <div class="content">
            <p>Hola <strong>{user_name}</strong>,</p>
            <p>¡Excelentes noticias! Tu solicitud de visita ha sido aceptada por el vendedor.</p>
            
            <div class="details">
                <h3 style="margin-top: 0; color: #2196F3;">Detalles de la Reserva</h3>
                <div class="detail-item">
                    <span class="detail-label">ID de Solicitud:</span> {request_id}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Propiedad:</span> {property_url}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Monto:</span> ${amount:,.2f}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Estado:</span> <span class="status-badge">ACEPTADA</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Fecha:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            
            <p>El monto de <strong>${amount:,.2f}</strong> ha sido debitado de tu wallet.</p>
            <p>Puedes descargar tu comprobante de pago desde tu historial de compras.</p>
            
            <p style="margin-top: 20px; font-size: 18px; color: #2196F3;"><strong>¡Disfruta tu visita!</strong></p>
        </div>
        <div class="footer">
            <p>Este es un correo automático, por favor no respondas a este mensaje.</p>
            <p>&copy; {datetime.now().year} Sistema de Propiedades</p>
        </div>
    </div>
</body>
</html>
        """
        
        return self.send_email(to_email, subject, body_text.strip(), body_html)
    
    def send_payment_rejected(
        self,
        to_email: str,
        user_name: str,
        request_id: str,
        property_url: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Enviar correo cuando el pago es rechazado.
        
        Args:
            to_email: Email del usuario
            user_name: Nombre del usuario
            request_id: ID de la solicitud
            property_url: URL de la propiedad
            reason: Razón del rechazo (opcional)
            
        Returns:
            bool: True si se envió exitosamente
        """
        subject = "❌ Solicitud Rechazada - Reserva de Visita"
        
        reason_text = f"\nRazón: {reason}" if reason else ""
        
        # Versión texto plano
        body_text = f"""
Hola {user_name},

Lamentamos informarte que tu solicitud de visita ha sido rechazada.

Detalles:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ID de Solicitud: {request_id}
• Propiedad: {property_url}
• Estado: RECHAZADA{reason_text}
• Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

No se ha realizado ningún cargo a tu wallet.
Puedes intentar reservar otra propiedad o contactar al vendedor para más información.

Saludos,
Sistema de Propiedades
        """
        
        # Versión HTML
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
        .content {{ background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
        .details {{ background-color: white; padding: 15px; margin: 15px 0; border-left: 4px solid #f44336; }}
        .detail-item {{ padding: 5px 0; }}
        .detail-label {{ font-weight: bold; color: #555; }}
        .footer {{ text-align: center; padding: 15px; color: #777; font-size: 12px; }}
        .icon {{ font-size: 48px; margin-bottom: 10px; }}
        .status-badge {{ background-color: #f44336; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">❌</div>
            <h1 style="margin: 0;">Solicitud Rechazada</h1>
        </div>
        <div class="content">
            <p>Hola <strong>{user_name}</strong>,</p>
            <p>Lamentamos informarte que tu solicitud de visita ha sido rechazada.</p>
            
            <div class="details">
                <h3 style="margin-top: 0; color: #f44336;">Detalles</h3>
                <div class="detail-item">
                    <span class="detail-label">ID de Solicitud:</span> {request_id}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Propiedad:</span> {property_url}
                </div>
                <div class="detail-item">
                    <span class="detail-label">Estado:</span> <span class="status-badge">RECHAZADA</span>
                </div>
                {f'<div class="detail-item"><span class="detail-label">Razón:</span> {reason}</div>' if reason else ''}
                <div class="detail-item">
                    <span class="detail-label">Fecha:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            
            <p>No se ha realizado ningún cargo a tu wallet.</p>
            <p>Puedes intentar reservar otra propiedad o contactar al vendedor para más información.</p>
        </div>
        <div class="footer">
            <p>Este es un correo automático, por favor no respondas a este mensaje.</p>
            <p>&copy; {datetime.now().year} Sistema de Propiedades</p>
        </div>
    </div>
</body>
</html>
        """
        
        return self.send_email(to_email, subject, body_text.strip(), body_html)

